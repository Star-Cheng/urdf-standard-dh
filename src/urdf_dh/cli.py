"""Command line interface and portable text exports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

from . import extract_standard_dh, extract_modified_dh


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def _nonnegative_integer(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urdf-dh",
        description="从 URDF 指定 base → tip 路径提取标准 DH 或 Craig MDH，并对比完整正运动学。",
    )
    parser.add_argument("input", type=Path, metavar="INPUT.urdf", help="已展开的 URDF 文件")
    parser.add_argument("--base", required=True, help="起点 link 名称")
    parser.add_argument("--tip", required=True, help="终点 link 名称")
    parser.add_argument("--convention", choices=("standard", "modified", "mdh"), default="standard",
                        help="standard=标准 DH（默认）；modified/mdh=Craig MDH")
    parser.add_argument("--output", type=Path, help="完整 JSON 输出；省略时写入 stdout")
    parser.add_argument("--csv", type=Path, help="DH 参数及变量映射 CSV")
    parser.add_argument("--markdown", type=Path, help="包含 DH 表和固定变换的 Markdown 报告")
    parser.add_argument("--samples", type=_positive_integer, default=100, help="随机验证样本数，默认 100")
    parser.add_argument("--seed", type=_nonnegative_integer, default=0, help="随机种子，默认 0")
    parser.add_argument("--position-tol", type=_positive_float, default=1e-7, help="位置误差容限，米，默认 1e-7")
    parser.add_argument("--rotation-tol", type=_positive_float, default=1e-7, help="旋转角误差容限，弧度，默认 1e-7")
    parser.add_argument("--parallel-tol", type=_positive_float, default=1e-12, help="平行轴判定的夹角正弦阈值，0 < 值 < 1，默认 1e-12")
    parser.add_argument("--no-validate", action="store_true", help="跳过 FK 采样验证，输出标记为未验证")
    return parser


def _paths(input_path: Path, candidates: Sequence[tuple[str, Path | None]]) -> list[tuple[str, Path]]:
    """Reject accidental input overwrites, output collisions and directories."""
    source = input_path.expanduser().resolve()
    result: list[tuple[str, Path]] = []
    for label, candidate in candidates:
        if candidate is None:
            continue
        path = candidate.expanduser().resolve()
        if path == source or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"{label} 输出不能覆盖输入文件：{candidate}")
        if path.is_dir():
            raise ValueError(f"{label} 输出路径是目录：{candidate}")
        for other_label, other in result:
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError(f"{label} 与 {other_label} 输出路径冲突：{candidate}")
        result.append((label, path))
    return result


def _number(value: float) -> str:
    return format(float(value), ".17g")


def _csv_text(model: Any) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    payload = model.to_dict()
    writer.writerow([
        "joint_name", "joint_type", "a_m", "alpha_rad", "d_offset_m",
        "theta_offset_rad", "source_name", "multiplier", "offset_rad_or_m",
        "traversal_direction_metadata", "geometry_case",
        "convention", "transform_order", "parameter_indexing", "frame_policy",
    ])
    for row in model.rows:
        writer.writerow([
            row.joint_name, row.joint_type, _number(row.a), _number(row.alpha),
            _number(row.d_offset), _number(row.theta_offset), row.source_name,
            _number(row.multiplier), _number(row.offset), _number(row.direction),
            row.geometry_case,
            model.convention, payload["transform_order"], payload["parameter_indexing"], model.frame_policy,
        ])
    return buffer.getvalue()


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _matrix_text(matrix: Any) -> str:
    return "\n".join("  ".join(_number(value) for value in row) for row in matrix)


def _markdown_text(model: Any, payload: dict[str, Any], base: str, tip: str) -> str:
    modified = model.convention == "modified"
    title = "MDH（Craig）" if modified else "标准 DH"
    a_label, alpha_label = ("a_(i-1)", "alpha_(i-1)") if modified else ("a_i", "alpha_i")
    formula = ("Rx(alpha_(i-1)) Tx(a_(i-1)) Rz(theta_i) Tz(d_i)" if modified
               else "Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)")
    policy_text = ("控制项目式，优先沿用 URDF 关节坐标方向，首轴轴向尺寸入表，法兰后 Tool 独立。"
                   if model.frame_policy == "controller" else "历史建系；使用原模型保存的 B/C 与关节映射。")
    lines = [
        f"# URDF {title} 提取报告", "",
        f"路径：{_cell(base)} → {_cell(tip)}。长度单位为米，角度单位为弧度。", "",
        f"约定：`A_i = {formula}`。", "",
        f"导出顺序：`{payload['transform_order']}`；参数索引：`{payload['parameter_indexing']}`。", "",
        "建系：" + policy_text, "",
        "完整正运动学：`T_base_tip(q) = B × A_1(q) × … × A_n(q) × C`。", "",
        "## DH 表", "",
        f"| i | 关节 | 类型 | {a_label} (m) | {alpha_label} (rad) | d_offset (m) | theta_offset (rad) | 轴线情况 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(model.rows, 1):
        lines.append("| " + " | ".join(map(_cell, [
            index, row.joint_name, row.joint_type, _number(row.a), _number(row.alpha),
            _number(row.d_offset), _number(row.theta_offset), row.geometry_case,
        ])) + " |")
    if not model.rows:
        lines.extend(["", "该路径没有活动关节，完整位姿由固定变换给出。"])
    lines.extend([
        "", "## 关节变量映射", "",
        "独立输入顺序：`" + ", ".join(str(name) for name in payload.get("input_names", [])) + "`。", "",
        "`q_DH = multiplier × q[source_name] + offset`。",
        "旋转关节：`theta = theta_offset + q_DH`，`d = d_offset`；移动关节：`d = d_offset + q_DH`，`theta = theta_offset`。",
        "`direction` 只记录 URDF 边的遍历方向；反向路径的运动方向已包含在 DH 轴中，不要再次乘上 direction。", "",
        "| 关节 | 输入 source_name | multiplier | offset (rad 或 m) | direction（元数据） |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in model.rows:
        lines.append("| " + " | ".join(map(_cell, [
            row.joint_name, row.source_name, _number(row.multiplier),
            _number(row.offset), _number(row.direction),
        ])) + " |")
    lines.extend([
        "", "## 基座固定变换 B", "", "```text", _matrix_text(model.base_transform), "```",
        "", "## 工具固定变换 C", "", "```text", _matrix_text(model.tool_transform), "```",
        "", "## 安装与法兰边界", "",
        "`B = mount_transform × base_alignment`；`C = tool_alignment × flange_to_tip`。",
        "边界轴向无法直接对齐时保留必要的固定对齐；不改变原始 URDF 几何精度。", "",
        "```json", json.dumps(payload.get("boundary_frames", {}), ensure_ascii=False, indent=2, allow_nan=False), "```",
        "", "## 验证结果", "",
        "以下为核心验证结果。采样验证检查 URDF 与 DH 模型的等价性，不代表实机标定精度。", "",
        "```json", json.dumps(payload.get("validation"), ensure_ascii=False, indent=2, allow_nan=False), "```",
        "", "## 几何容差", "",
        "平行轴阈值是夹角正弦的无量纲阈值；该几何近似仍受上述完整 FK 误差容限约束。", "",
        "```json", json.dumps(payload.get("geometry_tolerances"), ensure_ascii=False, indent=2, allow_nan=False), "```",
        "", "## 提示", "",
    ])
    warnings = payload.get("warnings", [])
    lines.extend(f"- {_cell(warning)}" for warning in warnings)
    if not warnings:
        lines.append("无额外提示。")
    lines.extend(["", "DH 建系一般不唯一；使用参数时应同时保留本报告中的 B、C 和关节映射。", ""])
    return "\n".join(lines)


def _write_atomic(path: Path, content: str) -> None:
    """Replace an output only after its UTF-8 contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 on success and 2 on invalid input, extraction or export errors."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        destinations = _paths(args.input, [
            ("JSON", args.output), ("CSV", args.csv), ("Markdown", args.markdown),
        ])
        extractor = extract_standard_dh if args.convention == "standard" else extract_modified_dh
        model = extractor(
            args.input.expanduser(), base_link=args.base, tip_link=args.tip,
            validate=not args.no_validate, samples=args.samples, seed=args.seed,
            position_tolerance=args.position_tol, rotation_tolerance=args.rotation_tol,
            parallel_tolerance=args.parallel_tol,
        )
        payload = model.to_dict()
        contents = {"JSON": json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
        if args.csv is not None:
            contents["CSV"] = _csv_text(model)
        if args.markdown is not None:
            contents["Markdown"] = _markdown_text(model, payload, args.base, args.tip)
        for label, path in destinations:
            _write_atomic(path, contents[label])
            print(f"已保存 {label}：{path}", file=sys.stderr)
        for warning in payload.get("warnings", []):
            print(f"提示：{warning}", file=sys.stderr)
        if args.output is None:
            sys.stdout.write(contents["JSON"])
        return 0
    except Exception as error:
        print(f"urdf-dh：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
