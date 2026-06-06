"""Tests for submission/configuration/config.py."""
import pytest

from ..submission.configuration import config as config_mod
from ..submission.configuration.config import (
    Configuration, ConfigOption, yml_join,
)
from ..submission.exc import ConfigurationError


def test_yml_join():
    class FakeLoader:
        def construct_sequence(self, node):
            return ["a", "b", "c"]
    assert yml_join(FakeLoader(), None) == "a/b/c"


def test_config_option_set_get():
    opt = ConfigOption({"x": 1, "y": 2})
    assert opt.x == 1
    assert opt.get("y") == 2
    assert opt.get("missing", "default") == "default"
    opt.set("x", 99)
    assert opt.x == 99 and opt.options["x"] == 99


def test_config_option_set_invalid_raises():
    opt = ConfigOption({"x": 1})
    with pytest.raises(ConfigurationError):
        opt.set("nope", 5)


def test_config_option_contains_and_str():
    opt = ConfigOption({"x": 1})
    assert "x" in opt
    assert "nope" not in opt
    assert "x=1" in str(opt)


def test_configuration_loads_default():
    cfg = Configuration()
    # paths is a nested ConfigOption with rhino_root / db_root etc.
    assert "paths" in cfg.options
    assert hasattr(cfg.paths, "rhino_root")


def test_configuration_parse_args_overrides_nested_path():
    cfg = Configuration()
    cfg.parse_args(["--path", "db_root=/tmp/somewhere"])
    assert cfg.paths.db_root == "/tmp/somewhere"


def test_configuration_parse_args_colon_separated():
    cfg = Configuration()
    cfg.parse_args(["--path", "db_root=/tmp/a:rhino_root=/tmp/b"])
    assert cfg.paths.db_root == "/tmp/a"
    assert cfg.paths.rhino_root == "/tmp/b"


def test_configuration_getattr_via_options():
    cfg = Configuration()
    # __getattr__ falls through to self.options for non-attribute names
    assert cfg.paths is cfg.options["paths"]


def test_configuration_str():
    cfg = Configuration()
    assert "paths" in str(cfg)
