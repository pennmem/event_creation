"""Tests for submission/viewers/recarray.py — recarray <-> dict/json
conversion and formatting helpers."""
import json

import numpy as np
import pytest

from ..submission.viewers import recarray


SIMPLE = [{"a": 1, "b": 2.5, "c": "hello"},
          {"a": 3, "b": 4.5, "c": "world"}]


def test_from_dict_and_to_dict_roundtrip():
    arr = recarray.from_dict(SIMPLE)
    assert arr.dtype.names == ("a", "b", "c")
    back = recarray.to_dict(arr)
    assert [e["a"] for e in back] == [1, 3]
    assert [e["c"] for e in back] == ["hello", "world"]


def test_from_dict_single_dict():
    arr = recarray.from_dict({"a": 1, "b": 2.0})
    assert len(arr) == 1 and arr[0]["a"] == 1


def test_from_dict_with_list_field():
    d = [{"x": 1, "vals": [1, 2, 3]}, {"x": 2, "vals": [4, 5]}]
    arr = recarray.from_dict(d)
    # padded to max length 3
    assert arr["vals"].shape == (2, 3)
    back = recarray.to_dict(arr)
    assert list(back[0]["vals"]) == [1, 2, 3]


def test_from_dict_with_nested_dict_field():
    d = [{"x": 1, "sub": {"p": 10, "q": 20}}]
    arr = recarray.from_dict(d)
    assert arr["sub"].dtype.names == ("p", "q")
    back = recarray.to_dict(arr)
    assert back[0]["sub"]["p"] == 10


def test_to_dict_zero_dim_returns_empty():
    arr = recarray.from_dict(SIMPLE)
    assert recarray.to_dict(arr[0]) == {}


def test_to_dict_respects_remove_flag():
    d = [{"a": 1, "_remove": False}, {"a": 2, "_remove": True}]
    arr = recarray.from_dict(d)
    back = recarray.to_dict(arr)
    # the entry with _remove=True is dropped, and _remove key is stripped
    assert len(back) == 1
    assert "_remove" not in back[0]
    assert back[0]["a"] == 1


def test_to_dict_all_removed_returns_empty_list():
    d = [{"a": 1, "_remove": True}, {"a": 2, "_remove": True}]
    arr = recarray.from_dict(d)
    assert recarray.to_dict(arr) == []


def test_list_of_dicts_field_roundtrip():
    # exercises mkdtype on list-of-dicts, copy_values list-of-dicts branch,
    # and to_dict on a size>1 nested recarray field
    d = [{"x": 1, "items": [{"p": 1}, {"p": 2}, {"p": 3}]}]
    arr = recarray.from_dict(d)
    assert arr["items"].dtype.names == ("p",)
    back = recarray.to_dict(arr)
    assert [e["p"] for e in back[0]["items"]] == [1, 2, 3]


def test_from_dict_empty_record_returns_empty_array():
    arr = recarray.from_dict([{}])
    assert len(arr) == 0


def test_to_json_string_and_file(tmp_path):
    arr = recarray.from_dict(SIMPLE)
    s = recarray.to_json(arr)
    loaded = json.loads(s)
    assert loaded[0]["a"] == 1
    out = tmp_path / "arr.json"
    with open(out, "w") as fp:
        recarray.to_json(arr, fp)
    assert json.loads(out.read_text())[0]["c"] == "hello"


def test_from_jsons_and_from_json(tmp_path):
    arr = recarray.from_jsons(json.dumps(SIMPLE))
    assert arr[0]["a"] == 1
    f = tmp_path / "in.json"
    f.write_text(json.dumps(SIMPLE))
    arr2 = recarray.from_json(str(f))
    assert arr2[1]["c"] == "world"


def test_from_json_old(tmp_path):
    f = tmp_path / "old.json"
    f.write_text(json.dumps({"a": 1, "b": 2.0}))
    arr = recarray.from_json_old(str(f))
    assert arr[0]["a"] == 1


def test_pformat_and_pprint_rec(capsys):
    arr = recarray.from_dict(SIMPLE)
    s = recarray.pformat_rec(arr)
    assert "a" in s and "b" in s and "c" in s
    recarray.pprint_rec(arr)
    assert "a" in capsys.readouterr().out


def test_pformat_rec_nested(capsys):
    d = [{"x": 1, "sub": {"p": 10, "q": 20}}]
    arr = recarray.from_dict(d)
    s = recarray.pformat_rec(arr[0])
    assert "sub" in s and "p" in s


def test_describe_recarray(capsys):
    arr = recarray.from_dict(SIMPLE)
    recarray.describe_recarray(arr)
    assert "a" in capsys.readouterr().out


def test_get_element_dtype_variants():
    assert recarray.get_element_dtype(1) == "int64"
    assert recarray.get_element_dtype(np.int32(1)) == "int64"
    assert recarray.get_element_dtype(1.5) == "float64"
    assert recarray.get_element_dtype("s") == "U256"
    assert recarray.get_element_dtype([1, 2, 3]) == "int64"
    dt = recarray.get_element_dtype({"a": 1})
    assert dt.names == ("a",)


def test_get_element_dtype_unknown_raises():
    with pytest.raises(Exception):
        recarray.get_element_dtype(object())


def test_mkdtype_from_list():
    dt = recarray.mkdtype([{"a": 1, "b": 2.0}])
    assert dt.names == ("a", "b")


def test_my_encoder_handles_numpy_and_bytes():
    enc = recarray.MyEncoder()
    assert enc.default(np.int64(5)) == 5
    assert enc.default(np.float64(2.5)) == 2.5
    assert enc.default(np.array([1, 2])) == [1, 2]
    assert enc.default(np.bool_(True)) is True
    assert enc.default(b"abc") == "abc"


def test_my_encoder_default_falls_through():
    enc = recarray.MyEncoder()
    with pytest.raises(TypeError):
        enc.default(object())


def test_strip_accents_deprecated(capsys):
    assert recarray.strip_accents("word") == "word"
    assert "deprecated" in capsys.readouterr().out
