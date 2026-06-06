"""Tests for submission/fileutil.py."""
import os
import stat

from ..submission import fileutil


def test_mkdir_creates_dir(tmp_path):
    target = tmp_path / "newdir"
    fileutil.mkdir(str(target))
    assert target.is_dir()


def test_makedirs_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    fileutil.makedirs(str(target))
    assert target.is_dir()


def test_open_with_perms_write_sets_644(tmp_path):
    f = tmp_path / "out.txt"
    with fileutil.open_with_perms(str(f), "w") as fh:
        fh.write("hello")
    assert f.read_text() == "hello"
    mode = stat.S_IMODE(os.stat(str(f)).st_mode)
    assert mode == 0o644


def test_open_with_perms_read_does_not_chmod(tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("data")
    os.chmod(str(f), 0o600)
    with fileutil.open_with_perms(str(f), "r") as fh:
        assert fh.read() == "data"
    # read mode leaves permissions untouched
    assert stat.S_IMODE(os.stat(str(f)).st_mode) == 0o600
