#!/usr/bin/env bash
# delete_ltp_cmlreader.sh
# Usage:
#   delete_ltp_cmlreader.sh <experiment> <subject> [session] [--dry-run|-n] [--log FILE] [--yes|-y]
#
# JSON schema:
#   protocols -> ltp -> subjects -> <SUBJECT> -> experiments -> <EXPERIMENT> -> sessions -> <SESSION>
#
# Behavior:
#   - If session is given: remove that session (JSON + folder), then prune empty ancestors.
#   - If session is omitted: confirm; remove ALL sessions for that experiment; then prune.
#   - Console output is minimal: only whether we found & deleted JSON/paths.
#   - All details (plans, JSON content, paths) are written to the log file.
#   - After a successful write, runs cp→rm→mv as RAM_maint so owner is RAM_maint, then chmod g+rw.
#
# Requirements: python3, sudo (for the owner-fix step)

set -euo pipefail

JSON="/protocols/ltp.json"
LTP_FOLDER="/protocols/ltp"

DRY_RUN=0
ASSUME_YES=0
LOG_FILE=""

usage() {
  cat <<'EOF'
Usage:
  delete_ltp_cmlreader.sh <experiment> <subject> [session] [--dry-run|-n] [--log FILE] [--yes|-y]

Options:
  -n, --dry-run     Show planned changes; do not modify JSON or delete files
      --log FILE    Append logs to FILE (default tries /var/log/... else ~/...)
  -y, --yes         Skip confirmation prompt when deleting ALL sessions
  -h, --help        Show this help
EOF
  exit "${1:-0}"
}

# --- Parse positional / help ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage 0; fi
EXPERIMENT="${1:-}"; shift || true
SUBJECT="${1:-}"; shift || true
SESSION=""
if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then SESSION="$1"; shift || true; fi
# --- Prompt for any missing required args (interactive only) ---
if [[ -z "$EXPERIMENT" || -z "$SUBJECT" ]]; then
  if [[ -t 0 ]]; then
    echo "Some required arguments are missing."
    # Ask only for what's missing; keep anything already provided
    while [[ -z "$EXPERIMENT" ]]; do
      read -r -p "Enter experiment name: " EXPERIMENT
      [[ -z "$EXPERIMENT" ]] && echo "Experiment cannot be empty."
    done
    while [[ -z "$SUBJECT" ]]; do
      read -r -p "Enter subject (e.g., LTP001): " SUBJECT
      [[ -z "$SUBJECT" ]] && echo "Subject cannot be empty."
    done
  else
    echo "Error: experiment and subject required (stdin not interactive)."
    usage 1
  fi
fi


# --- Parse flags ---
while (( "$#" )); do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -y|--yes)     ASSUME_YES=1; shift ;;
    --log)        LOG_FILE="${2:-}"; shift 2 ;;
    -h|--help)    usage 0 ;;
    *) echo "Unknown option: $1"; usage 1 ;;
  esac
done

SESSIONS_DIR="$LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT/sessions"

# --- Logging helpers ---
pick_log_file() {
  if [[ -n "$LOG_FILE" ]]; then echo "$LOG_FILE"; return; fi
  local sys="/var/log/delete_ltp_cmlreader.log"
  if { : >>"$sys"; } 2>/dev/null; then echo "$sys"; else echo "$HOME/delete_ltp_cmlreader.log"; fi
}
LOG_FILE="$(pick_log_file)"

log()  { printf '[%(%Y-%m-%d %H:%M:%S%z)T] %s\n' -1 "$*" >> "$LOG_FILE"; }
note() { echo "$*" | tee -a "$LOG_FILE" >/dev/null; }

backup_json() {
  local ts backup
  ts="$(date +"%Y%m%d-%H%M%S")"; backup="${JSON}.bak.${ts}"
  if (( DRY_RUN )); then
    echo "Backup: planned" ; log "Backup planned: $backup"
  else
    cp -p "$JSON" "$backup"
    echo "Backup: created"
    log "Backup created: $backup"
  fi
}

confirm_all() {
  if (( ASSUME_YES )); then return 0; fi
  read -r -p "Delete ALL sessions for subject='$SUBJECT', experiment='$EXPERIMENT'? Type YES to proceed: " ans
  case "${ans,,}" in y|yes) return 0 ;; YES) return 0 ;; *) return 1 ;; esac
}

# --- Owner fix via RAM_maint: cp→rm→mv + chmod g+rw ---
fix_owner_with_copy() {
  local json="${JSON:-/protocols/ltp.json}"
  if (( DRY_RUN )); then
    echo "Ownership: planned (RAM_maint)"
    log "OWNER-FIX PLAN: cp -p $json /protocols/ltp1.json; rm -f $json; mv -f /protocols/ltp1.json $json; chmod g+rw $json"
    return 0
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Ownership: skipped (no sudo) ⚠️"
    log "OWNER-FIX SKIPPED: sudo not available."
    return 0
  fi
  sudo -u RAM_maint bash -c '
    set -euo pipefail
    umask 007
    cp -p /protocols/ltp.json /protocols/ltp1.json
    rm -f /protocols/ltp.json
    mv -f /protocols/ltp1.json /protocols/ltp.json
    chmod g+rw /protocols/ltp.json
  '
  echo "Ownership: set to RAM_maint"
  log "OWNER-FIX DONE: $(ls -l "$json" 2>/dev/null || true)"
}

# ---------- Python helpers (schema-aware, no jq) ----------
py_show_single() {
python3 - "$JSON" "$SUBJECT" "$EXPERIMENT" "$SESSION" <<'PY'
import json, sys
path, subj, exp, sess = sys.argv[1:5]
with open(path) as f: d = json.load(f)
node = d.get("protocols", {}).get("ltp", {}).get("subjects", {}).get(subj, {}) \
       .get("experiments", {}).get(exp, {}).get("sessions", {})
val = node.get(str(sess))
print("__MISSING__" if val is None else json.dumps(val, indent=2, sort_keys=True))
PY
}

py_show_all_sessions() {
python3 - "$JSON" "$SUBJECT" "$EXPERIMENT" <<'PY'
import json, sys
path, subj, exp = sys.argv[1:4]
with open(path) as f: d = json.load(f)
node = d.get("protocols", {}).get("ltp", {}).get("subjects", {}).get(subj, {}) \
       .get("experiments", {}).get(exp, {}).get("sessions", None)
print("__MISSING__" if node is None else json.dumps(node, indent=2, sort_keys=True))
PY
}

# Delete one session, prune empties; plan|write (strict: errors if missing)
py_apply_single() {
python3 - "$JSON" "$SUBJECT" "$EXPERIMENT" "$SESSION" "$1" <<'PY'
import json, sys, tempfile, os, shutil, errno
path, subj, exp, sess, mode = sys.argv[1:6]
sess = str(sess)

def atomic_write(dst_path, data):
    d = os.path.dirname(dst_path) or "."
    fd, tmp = tempfile.mkstemp(prefix="ltp_json_", suffix=".tmp", dir=d)
    os.close(fd)
    with open(tmp, "w") as f: json.dump(data, f, indent=2, sort_keys=True)
    try:
        os.replace(tmp, dst_path)
    except OSError as e:
        if e.errno == errno.EXDEV:
            shutil.move(tmp, dst_path)
        else:
            raise

def plan_delete_single(d):
    removed = []
    p = d.get("protocols", {}).get("ltp", {}).get("subjects", {}).get(subj)
    if not p or "experiments" not in p or exp not in p["experiments"]:
        raise SystemExit("ERROR: subject/experiment not found in JSON")
    sessions = p["experiments"][exp].get("sessions", {})
    if sess not in sessions:
        raise SystemExit("ERROR: session not found in JSON")
    sessions.pop(sess, None)
    removed.append(f'json:protocols.ltp.subjects[{subj}].experiments[{exp}].sessions[{sess}]')
    eobj = p["experiments"][exp]
    if not sessions:
        other = {k:v for k,v in eobj.items() if k!="sessions" and v not in ({}, [], "", None)}
        if not other:
            p["experiments"].pop(exp, None)
            removed.append(f'json:protocols.ltp.subjects[{subj}].experiments[{exp}] (empty after prune)')
    if not p.get("experiments"):
        d["protocols"]["ltp"]["subjects"].pop(subj, None)
        removed.append(f'json:protocols.ltp.subjects[{subj}] (empty after prune)')
    return removed

with open(path) as f: data = json.load(f)
removed = plan_delete_single(data)
print("PLAN:"); [print(r) for r in removed]
if mode == "plan": sys.exit(0)
atomic_write(path, data)
print("WROTE:OK")
PY
}

# Delete all sessions, prune empties; tolerant if path missing; plan|write
py_apply_all_sessions() {
python3 - "$JSON" "$SUBJECT" "$EXPERIMENT" "$1" <<'PY'
import json, sys, tempfile, os, shutil, errno
path, subj, exp, mode = sys.argv[1:5]

def atomic_write(dst_path, data):
    d = os.path.dirname(dst_path) or "."
    fd, tmp = tempfile.mkstemp(prefix="ltp_json_", suffix=".tmp", dir=d)
    os.close(fd)
    with open(tmp, "w") as f: json.dump(data, f, indent=2, sort_keys=True)
    try:
        os.replace(tmp, dst_path)
    except OSError as e:
        if e.errno == errno.EXDEV:
            shutil.move(tmp, dst_path)
        else:
            raise

def plan_delete_all_sessions(d):
    removed = []
    prot = d.get("protocols", {})
    ltp  = prot.get("ltp", {})
    subs = ltp.get("subjects", {})
    subj_node = subs.get(subj)
    if not subj_node or "experiments" not in subj_node or exp not in subj_node["experiments"]:
        return removed, False
    exp_node = subj_node["experiments"][exp]
    exp_node["sessions"] = {}
    removed.append(f'json:protocols.ltp.subjects[{subj}].experiments[{exp}].sessions (cleared to {{}})')
    other = {k:v for k,v in exp_node.items() if k!="sessions" and v not in ({}, [], "", None)}
    if not exp_node.get("sessions") and not other:
        subj_node["experiments"].pop(exp, None)
        removed.append(f'json:protocols.ltp.subjects[{subj}].experiments[{exp}] (empty after prune)')
    if not subj_node.get("experiments"):
        subs.pop(subj, None)
        removed.append(f'json:protocols.ltp.subjects[{subj}] (empty after prune)')
    return removed, True

with open(path) as f: data = json.load(f)
removed, changed = plan_delete_all_sessions(data)
print("PLAN:"); [print(r) for r in removed] if removed else print("PLAN:(no JSON changes)")
if mode == "plan": sys.exit(0)
if changed:
    atomic_write(path, data)
    print("WROTE:OK")
else:
    print("WROTE:SKIPPED")
PY
}

# ---------- FS helpers (concise console; detail to log) ----------
print_fs_plan_single() {
  {
    echo "FS PLAN single:"
    local p="$SESSIONS_DIR/$SESSION"
    [[ -d "$p" ]] && echo "  delete $p" || echo "  missing $p"
    echo "  prune $SESSIONS_DIR"
    echo "  prune $LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT"
    echo "  prune $LTP_FOLDER/subjects/$SUBJECT"
  } >> "$LOG_FILE"
}

do_fs_single() {
  local p="$SESSIONS_DIR/$SESSION"
  if [[ -d "$p" ]]; then
    rm -rf -- "$p"
    echo "Folder deleted"
    log "Removed folder: $p"
  else
    echo "Folder not found"
    log "Folder missing (skipped): $p"
  fi
  rmdir "$SESSIONS_DIR" 2>/dev/null && log "Pruned empty: $SESSIONS_DIR" || true
  rmdir "$LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT" 2>/dev/null && log "Pruned empty: $LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT" || true
  rmdir "$LTP_FOLDER/subjects/$SUBJECT" 2>/dev/null && log "Pruned empty: $LTP_FOLDER/subjects/$SUBJECT" || true
}

print_fs_plan_all() {
  {
    echo "FS PLAN all:"
    if [[ -d "$SESSIONS_DIR" ]]; then
      echo "  delete-all under $SESSIONS_DIR"
      find "$SESSIONS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sed 's/^/    - /' | head -n 50
    else
      echo "  missing $SESSIONS_DIR"
    fi
    echo "  prune $SESSIONS_DIR"
    echo "  prune $LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT"
    echo "  prune $LTP_FOLDER/subjects/$SUBJECT"
  } >> "$LOG_FILE"
}

do_fs_all() {
  if [[ -d "$SESSIONS_DIR" ]]; then
    find "$SESSIONS_DIR" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | xargs -0 -I{} rm -rf -- "{}" || true
    echo "All session folders deleted"
    log "Removed all session folders under: $SESSIONS_DIR"
  else
    echo "No session folder to delete"
    log "Sessions directory not present: $SESSIONS_DIR"
  fi
  rmdir "$SESSIONS_DIR" 2>/dev/null && log "Pruned empty: $SESSIONS_DIR" || true
  rmdir "$LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT" 2>/dev/null && log "Pruned empty: $LTP_FOLDER/subjects/$SUBJECT/experiments/$EXPERIMENT" || true
  rmdir "$LTP_FOLDER/subjects/$SUBJECT" 2>/dev/null && log "Pruned empty: $LTP_FOLDER/subjects/$SUBJECT" || true
}

# ---------- Main ----------
echo "Deleting: experiment='$EXPERIMENT' subject='$SUBJECT' session='${SESSION:-ALL}'"
log  "----- delete_ltp_cmlreader start -----"
log  "Args: experiment='$EXPERIMENT' subject='$SUBJECT' session='${SESSION:-ALL}' dry_run=$DRY_RUN log='$LOG_FILE'"

if [[ ! -f "$JSON" ]]; then echo "JSON file missing"; log "ERROR: JSON not found at $JSON"; exit 1; fi

if [[ -n "$SESSION" ]]; then
  # Show minimal JSON status, log details
  if [[ "$(py_show_single)" == "__MISSING__" ]]; then
    echo "JSON entry not found"
  else
    echo "JSON entry exists"
  fi
  { echo "[DETAIL] Current JSON entry:"; py_show_single; } >> "$LOG_FILE"

  if (( DRY_RUN )); then
    py_apply_single plan >> "$LOG_FILE" 2>&1 && echo "Planned JSON delete" || { echo "Plan error"; log "Plan error (single)"; }
    print_fs_plan_single
    echo "Ownership: planned (RAM_maint)"
    log "DRY-RUN complete"
    exit 0
  fi

  backup_json
  if py_apply_single write >> "$LOG_FILE" 2>&1; then
    echo "JSON updated"
  else
    echo "JSON update failed"; exit 1
  fi
  do_fs_single
  fix_owner_with_copy
  echo "Done."
else
  if [[ "$(py_show_all_sessions)" == "__MISSING__" ]]; then
    echo "No JSON sessions found"
  else
    echo "JSON sessions found"
  fi
  { echo "[DETAIL] Current JSON sessions:"; py_show_all_sessions; } >> "$LOG_FILE"

  if (( DRY_RUN )); then
    py_apply_all_sessions plan >> "$LOG_FILE" 2>&1 && echo "Planned JSON delete-all" || { echo "Plan error"; log "Plan error (all)"; }
    print_fs_plan_all
    echo "Ownership: planned (RAM_maint)"
    log "DRY-RUN complete"
    exit 0
  fi

  if ! confirm_all; then echo "Aborted."; log "Aborted by user."; exit 1; fi

  backup_json
  if py_apply_all_sessions write >> "$LOG_FILE" 2>&1; then
    echo "JSON updated"
  else
    echo "JSON update skipped"; log "JSON write skipped (no JSON changes)"
  fi
  do_fs_all
  fix_owner_with_copy
  echo "Done."
fi

log "----- delete_ltp_cmlreader end -----"
