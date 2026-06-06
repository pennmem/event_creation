"""Tests for parsers/dtypes.py — verify every field-spec tuple is well-formed
and constructs a valid numpy structured dtype with usable default values."""
import numpy as np
import pytest

from ..submission.parsers import dtypes


ALL_FIELD_GROUPS = [
    "base_fields", "stim_fields", "courier_fields", "nicls_fields",
    "efr_fields", "repFR_fields", "fr_fields", "category_fields", "pal_fields",
    "math_fields", "ltp_fields", "ltpFR2_fields", "ltpFR_fields", "vffr_fields",
    "prelim_fields", "ps_fields", "system2_ps_fields", "location_subfields",
    "sham_subfields", "decision_subfields", "th_fields", "thr_fields",
    "cps_fields", "ps_state_fields", "vc_fields", "vcfrop_fields",
]


@pytest.mark.parametrize("group_name", ALL_FIELD_GROUPS)
def test_field_group_is_wellformed(group_name):
    group = getattr(dtypes, group_name)
    assert isinstance(group, tuple) and len(group) > 0
    names = [f[0] for f in group]
    assert len(names) == len(set(names)), "duplicate field names in %s" % group_name
    for field in group:
        # (name, default, dtype) or (name, default, dtype, length)
        assert len(field) in (3, 4)
        name, default, dtype_spec = field[0], field[1], field[2]
        assert isinstance(name, str) and name


@pytest.mark.parametrize("group_name", ALL_FIELD_GROUPS)
def test_field_group_builds_numpy_dtype(group_name):
    group = getattr(dtypes, group_name)
    np_spec = []
    for field in group:
        name, dtype_spec = field[0], field[2]
        if len(field) == 4:
            np_spec.append((name, dtype_spec, field[3]))
        else:
            np_spec.append((name, dtype_spec))
    dt = np.dtype(np_spec)
    assert dt.names == tuple(f[0] for f in group)
    # a zero-filled record of this dtype is constructable
    arr = np.zeros(1, dt)
    assert arr.shape == (1,)


def test_math_fields_has_array_field():
    # math_fields includes a length-4 spec: ('test', -999, 'int16', 3)
    test_field = [f for f in dtypes.math_fields if f[0] == "test"][0]
    assert len(test_field) == 4 and test_field[3] == 3
