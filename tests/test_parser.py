"""Analytic and malformed-input checks for the dependency-free URDF reader."""

import math

import numpy as np
import pytest

from urdf_dh.urdf import URDFError, URDFModel


TWO_R = """<robot name="test">
  <link name="base"/><link name="arm"/><link name="tip"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 2"/>
    <limit lower="-2" upper="2"/>
  </joint>
  <joint name="tool" type="fixed">
    <parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/>
  </joint>
</robot>"""


def test_forward_inverse_and_axis_geometry():
    model = URDFModel.from_xml(TWO_R)
    forward = model.chain("base", "tip")
    reverse = model.chain("tip", "base")
    expected = np.array(((0, -1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1)))
    np.testing.assert_allclose(forward.fk([math.pi / 2]), expected, atol=1e-14)
    np.testing.assert_allclose(reverse.fk({"shoulder": math.pi / 2}), np.linalg.inv(expected), atol=1e-14)
    assert forward.input_names == ("shoulder",)
    assert forward.input_limits == {"shoulder": (-2.0, 2.0)}
    assert [step.direction for step in reverse.steps] == [-1, -1]
    assert len(reverse.active_steps) == 1
    axes, zero_tip = forward.zero_geometry()
    np.testing.assert_array_equal(axes[0][0], [1, 0, 0])
    np.testing.assert_array_equal(axes[0][1], [0, 0, 1])
    np.testing.assert_array_equal(zero_tip[:3, 3], [2, 0, 0])
    axes, zero_tip = reverse.zero_geometry()
    np.testing.assert_array_equal(axes[0][0], [-1, 0, 0])
    np.testing.assert_array_equal(axes[0][1], [0, 0, -1])
    np.testing.assert_array_equal(zero_tip[:3, 3], [-2, 0, 0])


def test_rpy_order_joint_frame_and_prismatic_motion():
    model = URDFModel.from_xml("""<robot name="rpy">
      <link name="base"/><link name="tip"/>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
        <origin xyz="1 2 3" rpy="1.5707963267948966 1.5707963267948966 0"/>
        <axis xyz="0 -4 0"/>
      </joint></robot>""")
    # Ry(pi/2) Rx(pi/2) maps local -Y to global -X.
    expected = np.array(((0, 1, 0, 0.5), (0, 0, -1, 2), (-1, 0, 0, 3), (0, 0, 0, 1)))
    chain = model.chain("base", "tip")
    np.testing.assert_allclose(chain.fk([0.5]), expected, atol=1e-14)
    axes, _ = chain.zero_geometry()
    np.testing.assert_allclose(axes[0][1], [-1, 0, 0], atol=1e-14)


def test_branch_path_with_reverse_and_forward_steps():
    model = URDFModel.from_xml("""<robot name="branch">
      <link name="root"/><link name="left"/><link name="right"/>
      <joint name="turn" type="continuous"><parent link="root"/><child link="left"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="slide" type="prismatic"><parent link="root"/><child link="right"/>
        <origin xyz="0 2 0"/><axis xyz="0 1 0"/></joint>
      </robot>""")
    chain = model.chain("left", "right")
    assert [(step.joint.name, step.direction) for step in chain.steps] == [("turn", -1), ("slide", 1)]
    expected = np.array(((0, 1, 0, 2.5), (-1, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1)))
    np.testing.assert_allclose(chain.fk([math.pi / 2, 0.5]), expected, atol=1e-14)


def test_chained_mimic_with_external_source_and_ignored_geometric_offsets():
    # Declare dependents first to exercise cached mimic flattening as well.
    model = URDFModel.from_xml("""<robot name="mimic">
      <link name="root"/><link name="a"/><link name="b"/><link name="c"/>
      <joint name="third" type="continuous"><parent link="b"/><child link="c"/>
        <axis xyz="0 0 1"/><mimic joint="second" multiplier="-3" offset="0.2"/></joint>
      <joint name="second" type="revolute"><parent link="a"/><child link="b"/>
        <axis xyz="0 0 1"/><mimic joint="first" multiplier="2" offset="0.1"/></joint>
      <joint name="first" type="revolute"><parent link="root"/><child link="a"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
      </robot>""")
    chain = model.chain("b", "c")
    assert chain.input_names == ("first",)
    source, multiplier, offset = chain.joint_mapping("third")
    assert source == "first"
    assert multiplier == -6
    assert offset == pytest.approx(-0.1)
    assert chain.resolve_positions({"first": 0.25})["third"] == pytest.approx(-1.6)
    assert chain.resolve_positions()["third"] == pytest.approx(-0.1)
    _, geometric_zero = chain.zero_geometry()
    np.testing.assert_array_equal(geometric_zero, np.eye(4))
    assert not np.allclose(chain.fk(), geometric_zero)
    reversed_chain = model.chain("c", "b")
    np.testing.assert_allclose(reversed_chain.fk([0.25]), np.linalg.inv(chain.fk([0.25])), atol=1e-14)


def test_default_axis_fixed_only_and_same_link(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="fixed"><link name="a"/><link name="b"/>
      <joint name="f" type="fixed"><parent link="a"/><child link="b"/>
        <origin xyz="1 2 3"/></joint></robot>""", encoding="utf-8")
    model = URDFModel.from_file(path)
    np.testing.assert_array_equal(model.joints["f"].axis, [1, 0, 0])
    chain = model.chain("a", "b")
    assert chain.input_names == ()
    assert chain.input_limits == {}
    assert chain.active_steps == ()
    assert chain.resolve_positions([]) == {}
    np.testing.assert_array_equal(chain.fk()[:3, 3], [1, 2, 3])
    same = model.chain("b", "b")
    assert same.steps == ()
    assert same.zero_geometry()[0] == []
    np.testing.assert_array_equal(same.fk(), np.eye(4))


def test_unsupported_joint_only_rejected_when_selected():
    model = URDFModel.from_xml("""<robot name="floating"><link name="root"/><link name="arm"/>
      <link name="other"/><joint name="fixed" type="fixed"><parent link="root"/><child link="arm"/></joint>
      <joint name="float" type="floating"><parent link="root"/><child link="other"/></joint></robot>""")
    model.chain("root", "arm").fk()
    with pytest.raises(URDFError, match="unsupported.*floating"):
        model.chain("arm", "other")


def test_unused_zero_axis_from_cad_fixed_and_unselected_floating():
    model = URDFModel.from_xml("""<robot name="cad"><link name="root"/><link name="arm"/>
      <link name="other"/><joint name="fixed" type="fixed"><parent link="root"/><child link="arm"/>
        <origin xyz="1 2 3"/><axis xyz="0 0 0"/></joint>
      <joint name="float" type="floating"><parent link="root"/><child link="other"/>
        <axis xyz="0 0 0"/></joint></robot>""")
    np.testing.assert_array_equal(model.chain("root", "arm").fk()[:3, 3], [1, 2, 3])
    np.testing.assert_array_equal(model.joints["fixed"].axis, [1, 0, 0])
    np.testing.assert_array_equal(model.joints["float"].axis, [1, 0, 0])
    with pytest.raises(URDFError, match="unsupported.*floating"):
        model.chain("root", "other")


@pytest.mark.parametrize("kind", ["revolute", "continuous", "prismatic", "planar"])
def test_active_axis_types_still_reject_zero_axis(kind):
    xml = f'''<robot name="zero"><link name="a"/><link name="b"/>
      <joint name="j" type="{kind}"><parent link="a"/><child link="b"/>
        <axis xyz="0 0 0"/></joint></robot>'''
    with pytest.raises(URDFError, match="axis must be nonzero"):
        URDFModel.from_xml(xml)


@pytest.mark.parametrize("positions", [{"typo": 1}, [1, 2], [float("nan")], {"shoulder": float("inf")}, "0.2", 0.2, [[0.2]]])
def test_invalid_positions(positions):
    with pytest.raises(URDFError):
        URDFModel.from_xml(TWO_R).chain("base", "tip").fk(positions)


@pytest.mark.parametrize("replacement", [
    '<axis xyz="0 0 0"/>', '<axis xyz="0 nan 1"/>', '<axis xyz="0 inf 1"/>',
    '<axis xyz="0 1"/>', '<origin xyz="0 0 nan"/>', '<origin rpy="1 2 3 4"/>',
    '<limit lower="2" upper="-2"/>', '<limit effort="inf"/>',
])
def test_invalid_joint_numbers(replacement):
    xml = f'<robot name="r"><link name="a"/><link name="b"/><joint name="j" type="revolute"><parent link="a"/><child link="b"/>{replacement}</joint></robot>'
    with pytest.raises(URDFError):
        URDFModel.from_xml(xml)


@pytest.mark.parametrize("xml", [
    '<robot><link name="a"/><link name="a"/></robot>',
    '<robot><link name="a"/><link name="b"/></robot>',
    '<robot><link name="a"/><joint name="j" type="fixed"><parent link="a"/><child link="missing"/></joint></robot>',
    '<robot><link name="a"/><joint name="j" type="fixed"><parent link="a"/><child link="a"/></joint></robot>',
    '<robot><link name="a"/><link name="b"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="k" type="fixed"><parent link="a"/><child link="b"/></joint></robot>',
    '<robot><link name="a"/><link name="b"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="j" type="fixed"><parent link="b"/><child link="a"/></joint></robot>',
    '<robot><link name="a"/><link name="b"/><link name="c"/><joint name="j" type="fixed"><parent link="b"/><child link="c"/></joint><joint name="k" type="fixed"><parent link="c"/><child link="b"/></joint></robot>',
    '<robot><link name="a"/><link name="b"/><joint name="j" type="bogus"><parent link="a"/><child link="b"/></joint></robot>',
    '<robot><link name="a"/><link name="b"/><joint name="j" type="fixed"><child link="b"/></joint></robot>',
    '<robot/>', '<notrobot/>', '<robot>',
])
def test_invalid_structure(xml):
    with pytest.raises(URDFError):
        URDFModel.from_xml(xml)


@pytest.mark.parametrize("type1,type2,mimic1,mimic2", [
    ("revolute", "revolute", '<mimic joint="missing"/>', ''),
    ("revolute", "revolute", '<mimic joint="j2"/>', '<mimic joint="j1"/>'),
    ("revolute", "revolute", '<mimic joint="j1"/>', ''),
    ("revolute", "prismatic", '<mimic joint="j2"/>', ''),
    ("fixed", "revolute", '<mimic joint="j2"/>', ''),
    ("revolute", "fixed", '<mimic joint="j2"/>', ''),
    ("planar", "planar", '<mimic joint="j2"/>', ''),
    ("revolute", "revolute", '<mimic joint="j2" multiplier="nan"/>', ''),
])
def test_invalid_mimic(type1, type2, mimic1, mimic2):
    xml = f'''<robot name="m"><link name="a"/><link name="b"/><link name="c"/>
      <joint name="j1" type="{type1}"><parent link="a"/><child link="b"/>{mimic1}</joint>
      <joint name="j2" type="{type2}"><parent link="b"/><child link="c"/>{mimic2}</joint></robot>'''
    with pytest.raises(URDFError):
        URDFModel.from_xml(xml)


@pytest.mark.parametrize("xml", [
    '<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><xacro:macro name="thing"/></robot>',
    '<robot><link name="${link_name}"/></robot>',
    '<robot><link name="$(arg link_name)"/></robot>',
])
def test_unexpanded_xacro_reports_actionable_error(xml):
    with pytest.raises(URDFError, match="[Xx]acro"):
        URDFModel.from_xml(xml)


def test_large_axis_normalization_and_missing_file(tmp_path):
    model = URDFModel.from_xml(TWO_R.replace('axis xyz="0 0 2"', 'axis xyz="1e308 1e308 0"'))
    np.testing.assert_allclose(model.joints["shoulder"].axis, [1 / math.sqrt(2), 1 / math.sqrt(2), 0])
    with pytest.raises(URDFError, match="Cannot read"):
        URDFModel.from_file(tmp_path / "missing.urdf")
    with pytest.raises(URDFError, match="Unknown base"):
        model.chain("unknown", "tip")


@pytest.mark.parametrize("xml", [
    '<!DOCTYPE robot [<!ENTITY link_name "base">]><robot><link name="&link_name;"/></robot>',
    '<!DOCTYPE robot SYSTEM "file:///etc/hostname"><robot><link name="base"/></robot>',
    b'<!DOCTYPE robot><robot><link name="base"/></robot>',
])
def test_document_type_and_entities_are_rejected_before_parsing(xml):
    with pytest.raises(URDFError, match="DOCTYPE and ENTITY"):
        URDFModel.from_xml(xml)


def test_deep_tree_and_mimic_chain_without_recursion():
    count = 1100
    links = ''.join(f'<link name="l{i}"/>' for i in range(count + 1))
    joints = ''.join(f'<joint name="j{i}" type="revolute"><parent link="l{i}"/><child link="l{i+1}"/>'
                     + (f'<mimic joint="j{i+1}"/>' if i + 1 < count else '') + '</joint>'
                     for i in range(count))
    model = URDFModel.from_xml(f'<robot>{links}{joints}</robot>')
    chain = model.chain("l0", f"l{count}")
    assert len(chain.active_steps) == count
    assert chain.input_names == (f"j{count - 1}",)
    np.testing.assert_array_equal(chain.fk(), np.eye(4))
