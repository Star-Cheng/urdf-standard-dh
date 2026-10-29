"""Controller-style row gauges checked against independent physical URDF frames."""

import csv
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from urdf_dh import DHModel, DHRow, URDFModel, ValidationError, extract_dh
from urdf_dh.gui_controller import export_result

from .helpers import stress_robot, transform


def controller_robot(path, *, rounded=False, mount=False):
    half = 1.5708 if rounded else np.pi / 2
    full = 3.1416 if rounded else np.pi
    root = ET.Element("robot", name="physical_e1_controller_fixture")
    ET.SubElement(root, "link", name="root")
    ET.SubElement(root, "link", name="arm_base_link")
    node = ET.SubElement(root, "joint", name="mount", type="fixed")
    ET.SubElement(node, "parent", link="root")
    ET.SubElement(node, "child", link="arm_base_link")
    ET.SubElement(node, "origin", xyz="0.1 -0.2 0.3" if mount else "0 0 0",
                  rpy="0.2 -0.1 0.4" if mount else "0 0 0")
    xyzs = [(0, 0, .1745), (0, 0, 0), (0, .287, 0), (.018, 0, 0),
            (.018, -.314, 0), (0, 0, 0), (0, 0, 0)]
    rpys = [(0, 0, 0), (half, 0, 0), (-half, 0, 0), (-half, 0, full),
            (-half, 0, full), (half, -half, 0), (half, -half, 0)]
    previous = "arm_base_link"
    for index, (xyz, rpy) in enumerate(zip(xyzs, rpys), 1):
        child = f"link{index}"
        ET.SubElement(root, "link", name=child)
        node = ET.SubElement(root, "joint", name=f"q{index}", type="revolute")
        ET.SubElement(node, "parent", link=previous)
        ET.SubElement(node, "child", link=child)
        ET.SubElement(node, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
        ET.SubElement(node, "axis", xyz="0 0 1")
        ET.SubElement(node, "limit", lower="-2", upper="2", effort="1", velocity="1")
        previous = child
    for name, xyz, rpy in [("tcp1", "-0.0083277 -0.17809 0", f"{half} {half} 0"),
                           ("tcp2", "0 -0.301 0", f"{np.pi / 2} 0 0")]:
        ET.SubElement(root, "link", name=name)
        node = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(node, "parent", link="link7")
        ET.SubElement(node, "child", link=name)
        ET.SubElement(node, "origin", xyz=xyz, rpy=rpy)
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path


@pytest.mark.parametrize("convention", ["standard", "modified"])
@pytest.mark.parametrize("rounded", [False, True])
def test_first_joint_length_is_in_table_and_tool_is_physical(convention, rounded, tmp_path):
    path = controller_robot(tmp_path / "robot.urdf", rounded=rounded)
    model = extract_dh(path, "arm_base_link", "tcp1", convention=convention, samples=20)
    assert model.frame_policy == "controller"
    np.testing.assert_allclose(model.base_transform, np.eye(4), atol=1e-13, rtol=0)
    assert model.rows[0].d_offset == pytest.approx(.1745, abs=1e-14)
    half = 1.5708 if rounded else np.pi / 2
    expected_tool = transform((-.0083277, -.17809, 0), (half, half, 0))
    np.testing.assert_allclose(model.tool_transform, expected_tool, atol=1e-13, rtol=0)
    assert model.boundary_frames["mount_link"] == "arm_base_link"
    assert model.boundary_frames["flange_link"] == "link7"


def test_exact_controller_mdh_recovers_the_nominal_seven_axis_table(tmp_path):
    path = controller_robot(tmp_path / "robot.urdf")
    model = extract_dh(path, "arm_base_link", "link7", convention="modified")
    actual = np.array([[r.a, r.alpha, r.d_offset, r.theta_offset] for r in model.rows])
    h = np.pi / 2
    expected = np.array([[0, 0, .1745, 0], [0, h, 0, 0], [0, -h, .287, 0],
                         [.018, h, 0, np.pi], [.018, h, .314, np.pi],
                         [0, h, 0, h], [0, h, 0, h]])
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    np.testing.assert_allclose(model.tool_transform, np.eye(4), atol=1e-13, rtol=0)


@pytest.mark.parametrize("convention", ["standard", "modified"])
def test_table_independent_of_tcp_and_uses_explicit_installation(convention, tmp_path):
    path = controller_robot(tmp_path / "robot.urdf", mount=True, rounded=True)
    models = [extract_dh(path, "root", tip, convention=convention, samples=5)
              for tip in ("link7", "tcp1", "tcp2")]
    for model in models:
        assert model.rows == models[0].rows
        np.testing.assert_allclose(model.base_transform,
                                   transform((.1, -.2, .3), (.2, -.1, .4)), atol=1e-13, rtol=0)
        data = model.boundary_frames
        np.testing.assert_allclose(np.array(data["mount_transform"]) @ data["base_alignment"],
                                   model.base_transform, atol=1e-13, rtol=0)
        np.testing.assert_allclose(np.array(data["tool_alignment"]) @ data["flange_to_tip"],
                                   model.tool_transform, atol=1e-13, rtol=0)
        q = [.3, -.6, .8, -1.2, .5, -.2, .4]
        np.testing.assert_allclose(model.fk(q), URDFModel.from_file(path).chain("root", model.tip_link).fk(q),
                                   atol=1e-12, rtol=0)


def test_both_formulas_share_boundaries_but_have_distinct_link_parameters(tmp_path):
    path = controller_robot(tmp_path / "robot.urdf", rounded=True)
    standard = extract_dh(path, "arm_base_link", "tcp1", convention="standard")
    modified = extract_dh(path, "arm_base_link", "tcp1", convention="modified")
    np.testing.assert_allclose(standard.base_transform, modified.base_transform, atol=1e-13)
    np.testing.assert_allclose(standard.tool_transform, modified.tool_transform, atol=1e-13)
    assert standard.rows[0].alpha > 1.5
    assert modified.rows[0].alpha == 0
    for q in np.random.default_rng(14).uniform(-1, 1, (25, 7)):
        np.testing.assert_allclose(standard.fk(q), modified.fk(q), atol=1e-12, rtol=0)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
@pytest.mark.parametrize("reverse", [False, True])
def test_arbitrary_boundary_axes_preserve_signed_physical_inputs(tmp_path, kind, reverse):
    path = tmp_path / "arbitrary.urdf"
    path.write_text(f'''<robot name="arbitrary"><link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="q" type="{kind}"><parent link="base"/><child link="arm"/>
      <origin xyz="0.1 0.2 0.3" rpy="0.2 -0.4 0.7"/><axis xyz="1 2 3"/>
      <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="tool" type="fixed"><parent link="arm"/><child link="tip"/>
      <origin xyz="0.2 -0.1 0.4" rpy="-0.1 0.4 0.2"/></joint></robot>''')
    base, tip = ("tip", "base") if reverse else ("base", "tip")
    for convention in ("standard", "modified"):
        model = extract_dh(path, base, tip, convention=convention)
        restored = DHModel.from_dict(json.loads(json.dumps(model.to_dict())))
        for q in (-.7, 0, .8):
            expected = URDFModel.from_file(path).chain(base, tip).fk([q])
            np.testing.assert_allclose(restored.fk([q]), expected, atol=1e-12, rtol=0)
        assert model.boundary_frames
        assert any("alignment" in warning for warning in model.warnings)


@pytest.mark.parametrize("convention", ["standard", "modified"])
def test_legacy_json_keeps_its_original_gauge(convention):
    old = DHModel("base", "tip", (DHRow("q", "revolute", 0, 0, 0, 0, "q"),),
                  transform((0, 0, .1745)), np.eye(4), ("q",), convention=convention)
    document = old.to_dict()
    document.pop("frame_policy")
    document.pop("boundary_frames")
    if convention == "modified":
        document["transform_order"] = "Tx(a) Rx(alpha) Rz(theta) Tz(d)"
    loaded = DHModel.from_dict(document)
    assert loaded.frame_policy == "legacy"
    assert loaded.rows[0].d_offset == 0
    for q in (-.2, .3):
        np.testing.assert_allclose(loaded.fk([q]), old.fk([q]), atol=1e-14)


@pytest.mark.parametrize("field,value", [("frame_policy", "bad"), ("frame_policy", []),
                                       ("boundary_frames", []), ("boundary_frames", {}),
                                       ("transform_order", []), ("transform_order", None)])
def test_reject_invalid_frame_metadata(tmp_path, field, value):
    path = controller_robot(tmp_path / "robot.urdf")
    document = extract_dh(path, "arm_base_link", "tcp1").to_dict()
    document[field] = value
    with pytest.raises(ValueError):
        DHModel.from_dict(document)


@pytest.mark.parametrize("key", ["mount_transform", "base_alignment", "tool_alignment", "flange_to_tip"])
def test_reject_inconsistent_boundary_decomposition(tmp_path, key):
    path = controller_robot(tmp_path / "robot.urdf")
    document = extract_dh(path, "arm_base_link", "tcp1").to_dict()
    document["boundary_frames"][key][0][3] += .1
    with pytest.raises(ValueError, match="inconsistent"):
        DHModel.from_dict(document)


def test_exports_record_controller_policy_and_boundary_frames(tmp_path):
    path = controller_robot(tmp_path / "robot.urdf")
    model = extract_dh(path, "arm_base_link", "tcp1", convention="modified")
    assert json.loads(export_result(model, "json"))["frame_policy"] == "controller"
    record = next(csv.DictReader(io.StringIO(export_result(model, "csv"))))
    assert record["frame_policy"] == "controller"
    assert float(record["d_offset_m"]) == pytest.approx(.1745)
    report = export_result(model, "markdown")
    assert "控制项目式" in report and "flange_to_tip" in report


@pytest.mark.parametrize("convention", ["standard", "modified"])
def test_sampled_cancellation_cannot_hide_insufficient_numeric_resolution(tmp_path, convention):
    # Case 344 passed its original 13 poses (<1e-7 m), but an independent
    # configuration exceeded 1.7e-7 m. Its +/-1e9 m offsets have a larger
    # floating-point resolution estimate than the requested tolerance.
    robot, base, tip, _ = stress_robot(344)
    path = robot.write(tmp_path / "ill_conditioned.urdf")
    with pytest.raises(ValidationError, match="resolution estimate") as failure:
        extract_dh(path, base, tip, convention=convention, samples=12, seed=344)
    report = failure.value.report
    assert report["position_resolution_estimate"] > report["position_tolerance"]
    model = extract_dh(path, base, tip, convention=convention, samples=12, seed=344,
                       position_tolerance=1e-5)
    assert model.validation["passed"]

