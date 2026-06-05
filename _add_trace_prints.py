#!/usr/bin/env python3
"""One-off helper: insert function-entry trace prints into the event_creation
session-processing path. Idempotent. Records every insertion in a CSV manifest
so the change can be audited/undone.

Undo (authoritative): strip every line tagged with the marker:
    grep -rl "TRACE_AUTO_INSERTED" event_creation/submission \
      | xargs sed -i '/# TRACE_AUTO_INSERTED/d'
    rm trace_prints_manifest.csv
"""
import ast
import csv
import os

REPO = os.path.dirname(os.path.abspath(__file__))
MARKER = "# TRACE_AUTO_INSERTED"
MANIFEST = os.path.join(REPO, "trace_prints_manifest.csv")

FILES = [
    "event_creation/submission/pipelines.py",
    "event_creation/submission/tasks.py",
    "event_creation/submission/events_tasks.py",
    "event_creation/submission/transferer.py",
    "event_creation/submission/alignment/system1.py",
    "event_creation/submission/alignment/system2.py",
    "event_creation/submission/alignment/system3.py",
    "event_creation/submission/alignment/system4.py",
    "event_creation/submission/alignment/FreiburgAligner.py",
    "event_creation/submission/alignment/LTPAligner.py",
]


def docstring_end_line(node):
    """Return the last line number of the function's docstring, or None."""
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(
        getattr(body[0], "value", None), ast.Constant
    ) and isinstance(body[0].value.value, str):
        return body[0].end_lineno
    return None


def collect(tree, module):
    """Yield (insert_after_line, indent_col, classname, funcname, def_line)
    for every function/method. insert_after_line is the source line number
    *after* which the trace print should be inserted (1-based)."""
    out = []

    def walk(node, classname):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ds_end = docstring_end_line(child)
                if ds_end is not None:
                    after = ds_end
                else:
                    after = child.body[0].lineno - 1
                # indentation = column of first body statement
                indent = child.body[0].col_offset
                out.append((after, indent, classname, child.name, child.lineno))
                # nested functions too
                walk(child, classname)
            else:
                walk(child, classname)

    walk(tree, "module")
    return out


def already_traced(lines, after_idx):
    """Check if the line right after `after_idx` (0-based source index) is a marker."""
    nxt = after_idx  # the line we'd insert before, 0-based == after (1-based) line
    if nxt < len(lines) and MARKER in lines[nxt]:
        return True
    return False


def main():
    rows = []
    for rel in FILES:
        path = os.path.join(REPO, rel)
        with open(path, "r") as f:
            src = f.read()
        lines = src.splitlines(keepends=True)
        module = os.path.splitext(os.path.basename(rel))[0]
        tree = ast.parse(src)
        items = collect(tree, module)
        # sort bottom-to-top so insertions don't shift later indices
        items.sort(key=lambda t: t[0], reverse=True)
        for after, indent, classname, funcname, def_line in items:
            if already_traced(lines, after):
                continue
            msg = ">>> TRACE event_creation: %s.%s.%s (def L%d)" % (
                module, classname, funcname, def_line,
            )
            stmt = "%sprint(\"%s\")  %s\n" % (" " * indent, msg, MARKER)
            lines.insert(after, stmt)  # insert before 0-based index `after` == after 1-based line `after`
            rows.append({
                "file": rel,
                "class": classname,
                "function": funcname,
                "original_def_line": def_line,
                "print_statement": stmt.rstrip("\n"),
            })
        with open(path, "w") as f:
            f.writelines(lines)
        print("instrumented %s (%d functions)" % (rel, sum(1 for r in rows if r["file"] == rel)))

    # sort manifest for readability: by file then def line
    rows.sort(key=lambda r: (r["file"], r["original_def_line"]))
    write_header = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "class", "function", "original_def_line", "print_statement",
        ])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), MANIFEST))


if __name__ == "__main__":
    main()
