"""File selection, extraction and export workflow, independent of a GUI toolkit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

from . import DHModel, URDFError, URDFModel, extract_standard_dh, extract_modified_dh
from .geometry import normalize_convention
from .cli import _csv_text, _markdown_text, _paths, _write_atomic


MAX_VALIDATION_SAMPLES = 100_000


@dataclass(frozen=True)
class Document:
    """A selected file and its parsed snapshot, with suggested chain endpoints."""

    path: Path
    model: URDFModel
    links: tuple[str, ...]
    base_link: str
    tip_link: str
    fingerprint: str


def _file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise URDFError(f"无法读取 URDF 文件 {path}：{error}") from error


def _default_tip(model: URDFModel) -> str:
    """Prefer the leaf with most moving joints; break ties by the link name.

    Fixed frame depth does not increase a branch's priority. No chain is
    selected here, so a model containing planar/floating branches can still
    be loaded and the user can choose a supported path later.
    """
    children: dict[str, list[tuple[str, bool]]] = {name: [] for name in model.links}
    for joint in model.joints.values():
        children[joint.parent].append((joint.child, joint.type != "fixed"))
    leaves: list[tuple[int, str]] = []
    pending = [(model.root_link, 0)]
    while pending:
        name, moving = pending.pop()
        if not children[name]:
            leaves.append((moving, name))
        else:
            pending.extend((child, moving + int(active)) for child, active in children[name])
    return min(leaves, key=lambda item: (-item[0], item[1]))[1]


def load_document(path: str | Path) -> Document:
    """Read once, then parse and fingerprint the same bytes of expanded URDF."""
    selected = Path(path).expanduser().resolve()
    raw = _file_bytes(selected)
    model = URDFModel.from_xml(raw)
    return Document(
        path=selected,
        model=model,
        links=tuple(sorted(model.links)),
        base_link=model.root_link,
        tip_link=_default_tip(model),
        fingerprint=hashlib.sha256(raw).hexdigest(),
    )


def _checked_options(
    samples: int,
    seed: int,
    position_tolerance: float,
    rotation_tolerance: float,
    parallel_tolerance: float,
) -> dict[str, int | float]:
    if (isinstance(samples, bool) or not isinstance(samples, Integral)
            or not 1 <= samples <= MAX_VALIDATION_SAMPLES):
        raise ValueError(f"验证样本数必须是 1 到 {MAX_VALIDATION_SAMPLES} 之间的整数")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("随机种子必须是非负整数")
    for name, value in (
        ("位置容差", position_tolerance),
        ("旋转容差", rotation_tolerance),
        ("平行轴阈值", parallel_tolerance),
    ):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name}必须是有限正数")
    if parallel_tolerance >= 1:
        raise ValueError("平行轴阈值必须小于 1")
    return {
        "samples": int(samples),
        "seed": int(seed),
        "position_tolerance": float(position_tolerance),
        "rotation_tolerance": float(rotation_tolerance),
        "parallel_tolerance": float(parallel_tolerance),
    }


def parse_options(
    samples: str,
    seed: str,
    position_tol: str,
    rotation_tol: str,
    parallel_tol: str,
) -> dict[str, int | float]:
    """Parse editable UI fields into validated extraction keyword arguments."""
    integers = []
    for name, value in (("验证样本数", samples), ("随机种子", seed)):
        if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value.strip()) is None:
            raise ValueError(f"{name}必须填写非负整数")
        try:
            integers.append(int(value.strip()))
        except ValueError as error:
            raise ValueError(f"{name}不是有效整数") from error
    tolerances = []
    for name, value in (("位置容差", position_tol), ("旋转容差", rotation_tol), ("平行轴阈值", parallel_tol)):
        try:
            if not isinstance(value, str) or "_" in value:
                raise ValueError("not a numeric field")
            tolerances.append(float(value.strip()))
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{name}必须是有限正数") from error
    return _checked_options(*integers, *tolerances)


def convert_document(
    document: Document,
    base_link: str,
    tip_link: str,
    *,
    convention: str = "standard",
    samples: int = 100,
    seed: int = 0,
    position_tolerance: float = 1e-7,
    rotation_tolerance: float = 1e-7,
    parallel_tolerance: float = 1e-12,
) -> DHModel:
    """Extract and verify the selected path using the loaded file snapshot.

    A changed source is rejected instead of silently using a stale model.
    The path and hash stored in the result always describe the loaded bytes.
    """
    convention = normalize_convention(convention)
    options = _checked_options(samples, seed, position_tolerance, rotation_tolerance, parallel_tolerance)
    try:
        current = _file_bytes(document.path)
    except URDFError as error:
        raise URDFError(f"源文件无法读取，请重新选择或加载文件。{error}") from error
    if hashlib.sha256(current).hexdigest() != document.fingerprint:
        raise URDFError("源文件在加载后已改变，请重新加载文件再提取。")
    extractor = extract_standard_dh if convention == "standard" else extract_modified_dh
    result = extractor(
        document.model, base_link=base_link, tip_link=tip_link,
        validate=True, **options,
    )
    result.source_urdf = str(document.path)
    result.source_sha256 = document.fingerprint
    return result


def _format_name(format: str) -> str:
    if not isinstance(format, str) or format.strip().lower() not in {"json", "csv", "markdown"}:
        raise ValueError("导出格式必须是 json、csv 或 markdown")
    return format.strip().lower()


def export_result(model: DHModel, format: str) -> str:
    """Build a complete UTF-8 export without writing to disk or changing state."""
    kind = _format_name(format)
    payload: dict[str, Any] = model.to_dict()
    # Validate all payloads before export, including CSV fields and transforms
    # omitted from the table. Never save a NaN or partially serializable result.
    DHModel.from_dict(payload)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if kind == "json":
        return json_text
    if kind == "csv":
        return _csv_text(model)
    return _markdown_text(model, payload, model.base_link, model.tip_link)


def save_result(
    model: DHModel,
    path: str | Path,
    format: str,
    source_path: str | Path | None = None,
) -> Path:
    """Validate, protect the input file (including aliases), and save atomically."""
    content = export_result(model, format)
    destination = Path(path).expanduser().resolve()
    sources = [source for source in (source_path, model.source_urdf) if source is not None]
    if sources:
        for source in sources:
            _paths(Path(source), [(_format_name(format).upper(), destination)])
    elif destination.is_dir():
        raise ValueError(f"输出路径是目录：{path}")
    _write_atomic(destination, content)
    return destination
