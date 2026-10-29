"""GUI workflow contracts without importing Tk or requiring a display server."""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from urdf_dh import DHModel, URDFError
from urdf_dh import gui_controller as workflow


@pytest.fixture
def robot_path(tmp_path: Path) -> Path:
    path = tmp_path / "机械臂.urdf"
    path.write_text(
        """<robot name="gui_fixture">
          <link name="tip"/><link name="root"/><link name="arm"/>
          <joint name="joint1" type="continuous">
            <parent link="root"/><child link="arm"/>
            <origin xyz="0.1 0.2 0.3" rpy="0.2 -0.3 0.4"/>
            <axis xyz="0 0 1"/>
          </joint>
          <joint name="tool" type="fixed">
            <parent link="arm"/><child link="tip"/>
            <origin xyz="0.4 0.0 0.1" rpy="0.1 0.2 0.3"/>
          </joint>
        </robot>""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def model(robot_path: Path) -> DHModel:
    document = workflow.load_document(robot_path)
    return workflow.convert_document(document, "root", "tip", samples=4)


def _write_tree(path: Path, edges: list[tuple[str, str, str]]) -> Path:
    links = sorted({name for parent, child, _ in edges for name in (parent, child)}, reverse=True)
    xml = '<robot name="branch_fixture">'
    xml += "".join(f'<link name="{name}"/>' for name in links)
    xml += "".join(
        f'<joint name="j{index}" type="{kind}"><parent link="{parent}"/>'
        f'<child link="{child}"/><axis xyz="0 0 1"/></joint>'
        for index, (parent, child, kind) in enumerate(edges)
    )
    path.write_text(xml + "</robot>", encoding="utf-8")
    return path


def test_load_sorts_links_and_fingerprints_same_read(robot_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = robot_path.read_bytes()
    original = Path.read_bytes
    reads: list[Path] = []

    def record(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", record)
    document = workflow.load_document(robot_path)
    assert reads == [robot_path.resolve()]
    assert document.path == robot_path.resolve()
    assert document.links == ("arm", "root", "tip")
    assert (document.base_link, document.tip_link) == ("root", "tip")
    assert document.fingerprint == hashlib.sha256(raw).hexdigest()


def test_default_tip_prefers_moving_chain_over_deeper_fixed_branch(tmp_path: Path) -> None:
    source = _write_tree(tmp_path / "branches.urdf", [
        ("root", "a_fixed_1", "fixed"), ("a_fixed_1", "a_fixed_2", "fixed"),
        ("a_fixed_2", "a_fixed_3", "fixed"), ("a_fixed_3", "a_fixed_4", "fixed"),
        ("root", "b_1", "continuous"), ("b_1", "b_tip", "fixed"),
        ("root", "z_1", "continuous"), ("z_1", "z_2", "prismatic"),
        ("z_2", "z_tool", "fixed"),
    ])
    assert workflow.load_document(source).tip_link == "z_tool"


def test_default_tip_ties_are_deterministic_without_fixed_depth_bias(tmp_path: Path) -> None:
    edges = [
        ("root", "a_tip", "continuous"),
        ("root", "z_1", "continuous"), ("z_1", "z_2", "fixed"),
        ("z_2", "z_tip", "fixed"),
    ]
    forward = workflow.load_document(_write_tree(tmp_path / "forward.urdf", edges))
    reverse = workflow.load_document(_write_tree(tmp_path / "reverse.urdf", edges[::-1]))
    assert forward.tip_link == reverse.tip_link == "a_tip"


def test_unsupported_branch_does_not_prevent_loading_or_other_chain_conversion(tmp_path: Path) -> None:
    source = _write_tree(tmp_path / "unsupported.urdf", [
        ("root", "a_float", "floating"), ("a_float", "a_tip", "planar"),
        ("root", "z_supported", "continuous"),
    ])
    document = workflow.load_document(source)
    assert "a_float" in document.links
    result = workflow.convert_document(document, "root", "z_supported", samples=2)
    assert result.validation["passed"]
    with pytest.raises(URDFError, match="unsupported"):
        workflow.convert_document(document, "root", "a_tip", samples=2)


def test_single_link_load_and_conversion(tmp_path: Path) -> None:
    source = tmp_path / "single.urdf"
    source.write_text('<robot name="single"><link name="root"/></robot>', encoding="utf-8")
    document = workflow.load_document(source)
    assert (document.base_link, document.tip_link) == ("root", "root")
    result = workflow.convert_document(document, "root", "root", samples=1)
    assert not result.rows
    np.testing.assert_array_equal(result.fk(), np.eye(4))


def test_conversion_passes_snapshot_and_options(robot_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = workflow.load_document(robot_path)
    extract = workflow.extract_standard_dh
    calls = []

    def record(source: object, **options: object) -> DHModel:
        calls.append((source, options))
        return extract(source, **options)

    monkeypatch.setattr(workflow, "extract_standard_dh", record)
    result = workflow.convert_document(
        document, "tip", "root", samples=3, seed=42,
        position_tolerance=1e-6, rotation_tolerance=2e-6, parallel_tolerance=3e-13,
    )
    assert calls == [(document.model, {
        "base_link": "tip", "tip_link": "root", "validate": True,
        "samples": 3, "seed": 42, "position_tolerance": 1e-6,
        "rotation_tolerance": 2e-6, "parallel_tolerance": 3e-13,
    })]
    assert result.source_urdf == str(document.path)
    assert result.source_sha256 == document.fingerprint
    assert result.validation["passed"]
    np.testing.assert_allclose(result.fk([0.3]), document.model.chain("tip", "root").fk([0.3]), atol=1e-12)


def test_mimic_can_use_source_outside_selected_path(tmp_path: Path) -> None:
    source = tmp_path / "mimic.urdf"
    source.write_text("""<robot name="mimic">
      <link name="root"/><link name="driver_tip"/><link name="tip"/>
      <joint name="driver" type="continuous">
        <parent link="root"/><child link="driver_tip"/><axis xyz="0 0 1"/>
      </joint>
      <joint name="follower" type="continuous">
        <parent link="root"/><child link="tip"/><axis xyz="0 1 0"/>
        <origin xyz="0.1 0.2 0.3"/><mimic joint="driver" multiplier="-2" offset="0.3"/>
      </joint></robot>""", encoding="utf-8")
    document = workflow.load_document(source)
    result = workflow.convert_document(document, "root", "tip", samples=3)
    assert result.input_names == ("driver",)
    assert result.rows[0].multiplier == -2
    assert result.rows[0].offset == 0.3
    np.testing.assert_allclose(result.fk([0.7]), document.model.chain("root", "tip").fk([0.7]), atol=1e-12)


def test_changed_source_requires_reload_even_with_same_size_and_timestamp(robot_path: Path) -> None:
    document = workflow.load_document(robot_path)
    stat = robot_path.stat()
    robot_path.write_bytes(robot_path.read_bytes().replace(b"0.4 0.0 0.1", b"0.9 0.0 0.1"))
    os.utime(robot_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(URDFError, match="已改变.*重新加载"):
        workflow.convert_document(document, "root", "tip", samples=1)
    reloaded = workflow.load_document(robot_path)
    assert reloaded.fingerprint != document.fingerprint
    assert workflow.convert_document(reloaded, "root", "tip", samples=1).validation["passed"]


def test_deleted_source_requires_reload(robot_path: Path) -> None:
    document = workflow.load_document(robot_path)
    robot_path.unlink()
    with pytest.raises(URDFError, match="重新选择或加载"):
        workflow.convert_document(document, "root", "tip", samples=1)


def test_exports_reload_json_and_preserve_fk_and_traceability(model: DHModel) -> None:
    text = workflow.export_result(model, "json")
    restored = DHModel.from_dict(json.loads(text))
    assert restored.source_urdf == model.source_urdf
    assert restored.source_sha256 == model.source_sha256
    assert restored.validation == model.validation
    np.testing.assert_allclose(restored.fk([0.7]), model.fk([0.7]), atol=1e-12)
    rows = list(csv.DictReader(io.StringIO(workflow.export_result(model, "csv"))))
    assert rows[0]["joint_name"] == "joint1"
    assert float(rows[0]["a_m"]) == pytest.approx(model.rows[0].a)
    report = workflow.export_result(model, "markdown")
    for expected in ("root → tip", "基座固定变换 B", "工具固定变换 C", "source_name", "验证结果"):
        assert expected in report


@pytest.mark.parametrize("format,suffix", [("json", ".json"), ("csv", ".csv"), ("markdown", ".md")])
def test_saves_to_nested_path_atomically(model: DHModel, tmp_path: Path, format: str, suffix: str) -> None:
    path = tmp_path / "输出" / f"result{suffix}"
    assert workflow.save_result(model, path, format) == path.resolve()
    with path.open(encoding="utf-8", newline="") as handle:
        assert handle.read() == workflow.export_result(model, format)
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_save_protects_input_and_aliases(model: DHModel, robot_path: Path, tmp_path: Path, alias: str) -> None:
    destination = robot_path
    if alias != "direct":
        destination = tmp_path / f"{alias}.json"
        try:
            if alias == "symlink":
                destination.symlink_to(robot_path)
            else:
                os.link(robot_path, destination)
        except OSError as error:
            pytest.skip(str(error))
    original = robot_path.read_bytes()
    with pytest.raises(ValueError, match="不能覆盖输入文件"):
        workflow.save_result(model, destination, "json")
    assert robot_path.read_bytes() == original


def test_explicit_source_is_protected_when_model_has_no_metadata(model: DHModel, robot_path: Path) -> None:
    no_metadata = replace(model, source_urdf=None, source_sha256=None)
    with pytest.raises(ValueError, match="不能覆盖输入文件"):
        workflow.save_result(no_metadata, robot_path, "json", source_path=robot_path)


def test_explicit_source_does_not_disable_original_input_protection(model: DHModel, robot_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不能覆盖输入文件"):
        workflow.save_result(model, robot_path, "json", source_path=tmp_path / "other.urdf")


def test_invalid_export_does_not_create_directories_or_replace_file(model: DHModel, tmp_path: Path) -> None:
    broken = replace(model, rows=(replace(model.rows[0], a=float("nan")),))
    nested = tmp_path / "must_not_exist" / "bad.csv"
    with pytest.raises(ValueError, match="finite"):
        workflow.save_result(broken, nested, "csv")
    assert not nested.parent.exists()
    existing = tmp_path / "existing.csv"
    existing.write_text("keep original", encoding="utf-8")
    with pytest.raises(ValueError):
        workflow.save_result(broken, existing, "csv")
    assert existing.read_text(encoding="utf-8") == "keep original"


def test_bad_format_or_directory_is_rejected_without_writes(model: DHModel, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="导出格式"):
        workflow.save_result(model, tmp_path / "unused" / "result", "yaml")
    assert not (tmp_path / "unused").exists()
    with pytest.raises(ValueError, match="目录"):
        workflow.save_result(model, tmp_path, "json")


def test_parse_options_returns_core_keyword_arguments() -> None:
    assert workflow.parse_options(" 100 ", "42", "1e-7", "2e-7", "3e-12") == {
        "samples": 100, "seed": 42, "position_tolerance": 1e-7,
        "rotation_tolerance": 2e-7, "parallel_tolerance": 3e-12,
    }
    assert workflow.parse_options("100000", "0", "1", "1", "0.5")["samples"] == 100000


@pytest.mark.parametrize("index,value", [
    (0, ""), (0, "0"), (0, "100001"), (0, "1.5"), (0, "1e2"), (0, "1_0"),
    (1, "-1"), (1, "NaN"), (1, "1.0"),
    (2, "nan"), (2, "0"), (2, "-1"), (2, "1_0"), (2, "1e-999"),
    (3, "inf"), (3, "1e999"), (3, ""),
    (4, "nan"), (4, "0"), (4, "1"), (4, "2"),
])
def test_parse_options_rejects_invalid_input(index: int, value: str) -> None:
    values = ["100", "0", "1e-7", "1e-7", "1e-12"]
    values[index] = value
    with pytest.raises(ValueError):
        workflow.parse_options(*values)


@pytest.mark.parametrize("options", [{"samples": 0}, {"samples": True}, {"samples": 1.5}, {"seed": -1}, {"seed": True}, {"position_tolerance": False}])
def test_direct_conversion_cannot_bypass_numeric_validation(robot_path: Path, options: dict[str, object]) -> None:
    document = workflow.load_document(robot_path)
    with pytest.raises(ValueError):
        workflow.convert_document(document, "root", "tip", **options)
