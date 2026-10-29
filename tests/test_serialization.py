"""Portable model files must reject corrupt types, units and transforms."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from urdf_dh import DHModel, extract_standard_dh


@pytest.fixture
def document():
    path = Path(__file__).parents[1] / "examples" / "mixed_arm.urdf"
    return extract_standard_dh(path, "base_link", "tip_link", samples=5).to_dict()


@pytest.mark.parametrize("field,value", [
    ("multiplier", "1.0"), ("theta_offset", "0.0"), ("a", None),
    ("alpha", float("nan")), ("d_offset", float("inf")), ("offset", True),
    ("direction", True), ("direction", 0), ("joint_type", "floating"),
    ("source_name", "no_such_input"),
])
def test_reject_corrupt_rows(document, field, value):
    document["rows"][0][field] = value
    with pytest.raises(ValueError):
        DHModel.from_dict(document)


@pytest.mark.parametrize("field,value", [
    ("base_link", None), ("tip_link", ""), ("warnings", "abc"),
    ("rows", "abc"), ("input_names", "abc"), ("input_limits", []),
    ("units", {"length": "mm", "angle": "deg"}),
    ("convention", "modified"), ("convention", "unknown"), ("schema_version", 2),
    ("validation", "passed"),
])
def test_reject_corrupt_metadata(document, field, value):
    document[field] = value
    with pytest.raises(ValueError):
        DHModel.from_dict(document)


def test_reject_invalid_rotation(document):
    document["base_transform"][0][0] *= 2
    with pytest.raises(ValueError, match="rigid"):
        DHModel.from_dict(document)


def test_reject_improper_rotation(document):
    document["tool_transform"] = np.diag([-1., 1., 1., 1.]).tolist()
    with pytest.raises(ValueError, match="rigid"):
        DHModel.from_dict(document)


def test_reject_missing_fields(document):
    document.pop("base_transform")
    with pytest.raises(ValueError, match="Missing"):
        DHModel.from_dict(document)


def test_document_roundtrip_is_standalone(document):
    reloaded = DHModel.from_dict(json.loads(json.dumps(document, allow_nan=False)))
    inputs = {name: .35 for name in reloaded.input_names}
    expected = DHModel.from_dict(copy.deepcopy(document)).fk(inputs)
    np.testing.assert_allclose(reloaded.fk(inputs), expected, atol=1e-13)
    assert reloaded.to_dict() == document


def test_bad_joint_arguments(document):
    model = DHModel.from_dict(document)
    for positions in ({"typo": .1}, [0.], [[0, 0, 0]], [0, 0, float("nan")]):
        with pytest.raises(ValueError):
            model.fk(positions)
