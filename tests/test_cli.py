"""CLI contracts: usable exports, variable mapping, and safe file handling."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from urdf_dh import cli


@pytest.fixture
def robot_path(tmp_path: Path) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(
        """<robot name="cli_fixture">
          <link name="base"/><link name="arm"/><link name="tip"/>
          <joint name="joint1" type="continuous">
            <parent link="base"/><child link="arm"/>
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


def _args(path: Path) -> list[str]:
    return [str(path), "--base", "base", "--tip", "tip", "--samples", "5"]


def test_stdout_is_complete_json(robot_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(_args(robot_path)) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert len(result["rows"]) == 1
    assert result["input_names"] == ["joint1"]
    assert len(result["base_transform"]) == 4
    assert len(result["tool_transform"]) == 4
    assert result["validation"] is not None
    assert result["geometry_tolerances"]["parallel_axis_sine"] == pytest.approx(1e-12)


def test_exports_include_transforms_mapping_and_units(robot_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "nested" / "model.json"
    table = tmp_path / "nested" / "model.csv"
    report = tmp_path / "nested" / "model.md"
    assert cli.main(_args(robot_path) + ["--output", str(output), "--csv", str(table), "--markdown", str(report)]) == 0
    assert capsys.readouterr().out == ""
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["rows"]
    records = list(csv.DictReader(io.StringIO(table.read_text(encoding="utf-8"))))
    assert records[0]["joint_name"] == "joint1"
    assert float(records[0]["a_m"]) == pytest.approx(result["rows"][0]["a"])
    assert "alpha_rad" in records[0]
    assert "theta_offset_rad" in records[0]
    assert "traversal_direction_metadata" in records[0]
    text = report.read_text(encoding="utf-8")
    assert "Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)" in text
    assert "基座固定变换 B" in text
    assert "工具固定变换 C" in text
    assert "source_name" in text
    assert "不要再次乘上 direction" in text
    assert "验证结果" in text
    assert "几何容差" in text
    assert "parallel_axis_sine" in text


@pytest.mark.parametrize("option", ["--output", "--csv", "--markdown"])
def test_refuses_input_overwrite(robot_path: Path, option: str, capsys: pytest.CaptureFixture[str]) -> None:
    original = robot_path.read_bytes()
    assert cli.main(_args(robot_path) + [option, str(robot_path)]) == 2
    assert robot_path.read_bytes() == original
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "不能覆盖输入文件" in captured.err
    assert "Traceback" not in captured.err


def test_refuses_output_collision_before_creating_files(robot_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "new_directory" / "shared.txt"
    assert cli.main(_args(robot_path) + ["--output", str(target), "--csv", str(target)]) == 2
    assert not target.parent.exists()
    assert "路径冲突" in capsys.readouterr().err


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_refuses_input_alias(robot_path: Path, tmp_path: Path, link_type: str, capsys: pytest.CaptureFixture[str]) -> None:
    alias = tmp_path / "input_alias.json"
    try:
        if link_type == "symlink":
            alias.symlink_to(robot_path)
        else:
            os.link(robot_path, alias)
    except OSError as error:
        pytest.skip(f"filesystem does not support {link_type}: {error}")
    original = robot_path.read_bytes()
    assert cli.main(_args(robot_path) + ["--output", str(alias)]) == 2
    assert robot_path.read_bytes() == original
    assert "不能覆盖输入文件" in capsys.readouterr().err


def test_bad_urdf_is_error_without_output_or_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "broken.urdf"
    source.write_text("<robot>", encoding="utf-8")
    target = tmp_path / "new" / "result.json"
    assert cli.main(_args(source) + ["--output", str(target)]) == 2
    assert not target.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
    assert "Traceback" not in captured.err


def test_verification_failure_creates_no_output(robot_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def failure(*args: object, **kwargs: object) -> object:
        raise ValueError("DH / URDF FK validation failed")

    monkeypatch.setattr(cli, "extract_standard_dh", failure)
    directory = tmp_path / "must_not_be_created"
    assert cli.main(_args(robot_path) + [
        "--output", str(directory / "result.json"),
        "--csv", str(directory / "result.csv"),
        "--markdown", str(directory / "result.md"),
    ]) == 2
    assert not directory.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "validation failed" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("option,value", [
    ("--samples", "0"), ("--samples", "-1"), ("--seed", "-1"),
    ("--position-tol", "nan"), ("--rotation-tol", "inf"),
    ("--rotation-tol", "0"),
    ("--parallel-tol", "0"), ("--parallel-tol", "nan"),
    ("--parallel-tol", "1"), ("--parallel-tol", "2"),
])
def test_rejects_invalid_numeric_options(robot_path: Path, option: str, value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(_args(robot_path) + [option, value]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_forwards_verification_options(robot_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    original = cli.extract_standard_dh
    calls: list[dict[str, object]] = []

    def record(path: Path, **kwargs: object) -> object:
        calls.append(kwargs)
        return original(path, **kwargs)

    monkeypatch.setattr(cli, "extract_standard_dh", record)
    assert cli.main(_args(robot_path) + ["--seed", "42", "--position-tol", "1e-6", "--rotation-tol", "2e-6", "--parallel-tol", "5e-13", "--no-validate"]) == 0
    assert calls == [{
        "base_link": "base", "tip_link": "tip", "validate": False,
        "samples": 5, "seed": 42,
        "position_tolerance": 1e-6, "rotation_tolerance": 2e-6,
        "parallel_tolerance": 5e-13,
    }]
    assert "validation" in json.loads(capsys.readouterr().out)


def test_python_module_entrypoint(robot_path: Path) -> None:
    environment = os.environ.copy()
    source_directory = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source_directory + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.run(
        [sys.executable, "-m", "urdf_dh", *_args(robot_path)],
        env=environment, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["input_names"] == ["joint1"]
