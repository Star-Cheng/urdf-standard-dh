"""Independent Craig-MDH mechanisms, plus convention/serialization integration."""

import csv
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from urdf_dh import (DHModel, URDFModel, extract_dh, extract_modified_dh,
                     extract_standard_dh, mdh_matrix)
from urdf_dh import cli
from urdf_dh.gui_controller import convert_document, export_result, load_document

from .helpers import stress_robot, transform


def elementary_mdh(theta, d, a, alpha):
    """Independent elementary transforms, not the production closed form."""
    return (transform((a, 0, 0)) @ transform(rpy=(alpha, 0, 0))
            @ transform(rpy=(0, 0, theta)) @ transform((0, 0, d)))


def write_known_mdh(path, rows):
    """Construct URDF straight from Craig X_(i-1), Z_i; no DH converter used."""
    root = ET.Element("robot", name="independent_craig")
    ET.SubElement(root, "link", name="base")

    def joint(name, kind, parent, child, xyz, rpy):
        ET.SubElement(root, "link", name=child)
        node = ET.SubElement(root, "joint", name=name, type=kind)
        ET.SubElement(node, "parent", link=parent)
        ET.SubElement(node, "child", link=child)
        ET.SubElement(node, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
        if kind != "fixed":
            ET.SubElement(node, "axis", xyz="0 0 1")
            if kind != "continuous":
                ET.SubElement(node, "limit", lower="-1", upper="1", effort="1", velocity="1")

    joint("mount", "fixed", "base", "mount", (.12, -.23, .34), (.21, -.4, .1))
    previous = "mount"
    for i, (kind, a, alpha, d, theta) in enumerate(rows):
        joint(f"x{i}", "fixed", previous, f"pre{i}", (a, 0, 0), (alpha, 0, 0))
        joint(f"q{i}", kind, f"pre{i}", f"body{i}", (0, 0, d), (0, 0, theta))
        previous = f"body{i}"
    joint("tool", "fixed", previous, "tip", (-.03, .07, .11), (-.12, .3, .7))
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path


@pytest.mark.parametrize("seed", range(6))
def test_craig_matrix_against_elementary_transforms(seed):
    rng = np.random.default_rng(seed)
    for theta, d, a, alpha in rng.uniform(-2, 2, (20, 4)):
        np.testing.assert_allclose(mdh_matrix(theta, d, a, alpha),
                                   elementary_mdh(theta, d, a, alpha), atol=1e-14, rtol=0)


@pytest.mark.parametrize("kinds", [
    ("revolute",), ("prismatic",), ("revolute", "prismatic", "continuous"),
    ("prismatic", "prismatic", "revolute"), ("revolute", "revolute", "revolute"),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_extract_against_independently_built_craig_robot(tmp_path, kinds, reverse):
    rows = [(kind, .12 * (i + 1), [np.pi, -.7, 0][i % 3], -.13 + .05 * i, .2 - .3 * i)
            for i, kind in enumerate(kinds)]
    path = write_known_mdh(tmp_path / "craig.urdf", rows)
    base, tip = ("tip", "base") if reverse else ("base", "tip")
    model = extract_modified_dh(path, base, tip, samples=20)
    standard = extract_standard_dh(path, base, tip, samples=20)
    assert len(model.rows) == len(rows)
    assert model.rows[0].a == model.rows[0].alpha == 0
    restored = DHModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.to_dict() == model.to_dict()
    for qs in np.vstack([np.zeros(len(rows)), np.random.default_rng(12).uniform(-.7, .7, (30, len(rows)))]):
        inputs = dict(zip([f"q{i}" for i in range(len(rows))], qs))
        expected = transform((.12, -.23, .34), (.21, -.4, .1))
        for (kind, a, alpha, d, theta), q in zip(rows, qs):
            expected = expected @ elementary_mdh(theta + (q if kind != "prismatic" else 0),
                                                  d + (q if kind == "prismatic" else 0), a, alpha)
        expected = expected @ transform((-.03, .07, .11), (-.12, .3, .7))
        if reverse:
            expected = np.linalg.inv(expected)
        np.testing.assert_allclose(model.fk(inputs), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(restored.fk(inputs), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(standard.fk(inputs), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("case", [0, 8, 52, 99, 100, 147, 199, 400, 433, 455, 499])
def test_mdh_with_arbitrary_urdf_frames_degenerate_axes_and_reverse_mimic(tmp_path, case):
    robot, base, tip, _ = stress_robot(case)
    path = robot.write(tmp_path / "generated.urdf")
    model = extract_modified_dh(path, base, tip, samples=20)
    rng = np.random.default_rng(case)
    for _ in range(15):
        inputs = {name: float(rng.uniform(-.5, .5)) for name in model.input_names}
        np.testing.assert_allclose(model.fk(inputs), robot.fk(inputs, base, tip), atol=1e-8, rtol=0)


@pytest.mark.parametrize("base,tip", [("base_link", "base_link"), ("link_3", "tip_link"),
                                      ("tip_link", "link_3")])
def test_fixed_only_mdh_paths(base, tip):
    from pathlib import Path
    path = Path(__file__).parents[1] / "examples" / "mixed_arm.urdf"
    model = extract_modified_dh(path, base, tip)
    assert model.rows == ()
    np.testing.assert_allclose(model.fk(), URDFModel.from_file(path).chain(base, tip).fk(), atol=1e-12)


@pytest.mark.parametrize("convention", ["standard", "modified", "mdh"])
def test_api_cli_and_gui_controller_conventions(tmp_path, capsys, convention):
    path = write_known_mdh(tmp_path / "input.urdf", [("revolute", .2, .6, .1, -.2)])
    expected_convention = "modified" if convention == "mdh" else convention
    model = extract_dh(path, "base", "tip", convention=convention, samples=3)
    assert model.convention == expected_convention
    assert cli.main([str(path), "--base", "base", "--tip", "tip",
                     "--convention", convention, "--samples", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["convention"] == expected_convention
    converted = convert_document(load_document(path), "base", "tip", convention=convention, samples=3)
    for kind in ("json", "csv", "markdown"):
        exported = export_result(converted, kind)
        if kind == "json":
            assert json.loads(exported)["convention"] == expected_convention
        elif kind == "csv":
            records = list(csv.DictReader(io.StringIO(exported)))
            assert records[0]["convention"] == expected_convention
            assert records[0]["transform_order"] == payload["transform_order"]
        else:
            assert payload["transform_order"] in exported
            assert ("a_(i-1)" if expected_convention == "modified" else "a_i") in exported
    np.testing.assert_allclose(DHModel.from_dict(payload).fk([.4]), model.fk([.4]), atol=1e-12)


@pytest.mark.parametrize("convention", ["MDH", "unknown", "", None, 2, []])
def test_invalid_conventions_rejected(tmp_path, convention):
    with pytest.raises(ValueError, match="convention"):
        extract_dh(tmp_path / "no_file.urdf", "base", "tip", convention=convention)


def test_cannot_relabel_standard_document_as_modified(tmp_path):
    path = write_known_mdh(tmp_path / "input.urdf", [("revolute", .2, .6, .1, -.2)])
    document = extract_standard_dh(path, "base", "tip").to_dict()
    document["convention"] = "modified"
    with pytest.raises(ValueError, match="transform_order"):
        DHModel.from_dict(document)


def test_old_standard_json_without_indexing_still_loads(tmp_path):
    path = write_known_mdh(tmp_path / "input.urdf", [("prismatic", .2, .6, .1, -.2)])
    model = extract_standard_dh(path, "base", "tip")
    document = model.to_dict()
    document.pop("parameter_indexing")
    np.testing.assert_allclose(DHModel.from_dict(document).fk([.2]), model.fk([.2]), atol=1e-12)


@pytest.mark.parametrize("parameter", ["theta", "d", "a", "alpha"])
def test_mdh_matrix_rejects_nonfinite(parameter):
    values = dict(theta=0, d=0, a=0, alpha=0)
    values[parameter] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        mdh_matrix(**values)
