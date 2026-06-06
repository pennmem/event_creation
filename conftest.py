"""Repo-root pytest configuration.

Two things have to happen *before* any ``event_creation.submission`` module is
imported by a test module:

1. ``matplotlib`` must be switched to the headless ``Agg`` backend (several
   submission/heartbeat modules import ``pyplot`` at import time).
2. ``db_root`` must point at a writable directory. ``submission/log.py``
   instantiates ``logger = Logger()`` at import time, which opens
   ``<db_root>/protocols/log.txt``; with the default ``db_root='/'`` this
   raises ``PermissionError`` and leaves ``logger=None``. Overriding it to a
   temp directory lets the whole package import cleanly under test.

conftest.py at the repo root is imported by pytest before collecting any test
modules, so doing the override here covers every test file.
"""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

# Point db_root at a writable temp dir before submission modules are imported.
_TEST_DB_ROOT = tempfile.mkdtemp(prefix="event_creation_test_dbroot_")
os.makedirs(os.path.join(_TEST_DB_ROOT, "protocols"), exist_ok=True)

from event_creation.submission.configuration import config, paths  # noqa: E402

config.parse_args(["--path", "db_root={}".format(_TEST_DB_ROOT)])
assert paths.db_root == _TEST_DB_ROOT

# Re-create the module-level logger now that db_root is writable, so modules
# that grabbed ``log.logger`` (possibly None) get a working logger.
import importlib  # noqa: E402
from event_creation.submission import log as _log  # noqa: E402

importlib.reload(_log)
