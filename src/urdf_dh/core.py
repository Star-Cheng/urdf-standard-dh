"""Controller-style DH/MDH frames, signed inputs and explicit boundary transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import copy
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (TRANSFORM_ORDERS, dh_matrix, mdh_matrix, normalize_convention,
                       frame, inverse, perpendicular, pose_errors)
from .urdf import URDFModel


class ConversionError(ValueError):
    """A geometrically valid and numerically verified DH model could not be built."""


class ValidationError(ConversionError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        resolution_note = ""
        if report.get("position_resolution_estimate", 0) > report.get("position_tolerance", math.inf):
            resolution_note = (f"Geometric floating-point position resolution estimate "
                               f"{report['position_resolution_estimate']:.6g} m exceeds the "
                               f"requested {report['position_tolerance']:.6g} m tolerance. ")
        super().__init__(
            "DH / URDF FK validation failed: "
            f"position={report['max_position_error']:.6g} m, "
            f"rotation={report['max_rotation_error']:.6g} rad. "
            + resolution_note +
            "Near-parallel axes or very large geometry may be ill-conditioned; "
            "inspect warnings and coordinate precision."
        )


@dataclass(frozen=True)
class DHRow:
    joint_name: str
    joint_type: str
    a: float
    alpha: float
    d_offset: float
    theta_offset: float
    source_name: str
    multiplier: float = 1.0
    offset: float = 0.0
    direction: int = 1
    geometry_case: str = ""

    def transform(self, inputs: Mapping[str, float], *, convention: str = "standard") -> np.ndarray:
        matrix = dh_matrix if normalize_convention(convention) == "standard" else mdh_matrix
        coordinate = self.multiplier * inputs[self.source_name] + self.offset
        if not math.isfinite(coordinate):
            raise ValueError(f"Non-finite mapped coordinate for {self.joint_name!r}")
        if self.joint_type == "prismatic":
            return matrix(self.theta_offset, self.d_offset + coordinate, self.a, self.alpha)
        return matrix(self.theta_offset + coordinate, self.d_offset, self.a, self.alpha)


def _checked_transform(value: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = array[:3, :3]
    if (not np.allclose(array[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8)):
        raise ValueError(f"{name} must be a rigid homogeneous transform")
    return array


@dataclass
class DHModel:
    base_link: str
    tip_link: str
    rows: tuple[DHRow, ...]
    base_transform: np.ndarray
    tool_transform: np.ndarray
    input_names: tuple[str, ...]
    input_limits: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    validation: dict[str, Any] | None = None
    source_urdf: str | None = None
    source_sha256: str | None = None
    parallel_tolerance: float = 1e-12
    convention: str = "standard"
    frame_policy: str = "legacy"
    boundary_frames: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.convention = normalize_convention(self.convention)
        if self.frame_policy not in ("legacy", "controller"):
            raise ValueError("frame_policy must be 'legacy' or 'controller'")

    def _inputs(self, positions: Mapping[str, float] | Sequence[float] | None) -> dict[str, float]:
        if positions is None:
            result = dict.fromkeys(self.input_names, 0.0)
        elif isinstance(positions, Mapping):
            extra = set(positions) - set(self.input_names)
            if extra:
                raise ValueError(f"Unknown independent joint names: {sorted(extra)}")
            result = {name: float(positions.get(name, 0.0)) for name in self.input_names}
        else:
            values = np.asarray(positions, dtype=float)
            if values.shape != (len(self.input_names),):
                raise ValueError(f"Expected {len(self.input_names)} independent joint positions")
            result = dict(zip(self.input_names, values.tolist()))
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Joint positions must be finite")
        return result

    def fk(self, positions: Mapping[str, float] | Sequence[float] | None = None) -> np.ndarray:
        """Complete requested base->tip pose, including B and tool transforms."""
        inputs = self._inputs(positions)
        transform = self.base_transform.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                for row in self.rows:
                    transform = transform @ row.transform(inputs, convention=self.convention)
                transform = transform @ self.tool_transform
        except FloatingPointError as error:
            raise ValueError("DH forward kinematics overflowed; check geometry and joint positions") from error
        if not np.isfinite(transform).all():
            raise ValueError("DH forward kinematics must remain finite")
        return transform

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "convention": self.convention,
            "transform_order": TRANSFORM_ORDERS[self.convention],
            "parameter_indexing": ("a_i, alpha_i, d_i, theta_i" if self.convention == "standard"
                                   else "a_(i-1), alpha_(i-1), d_i, theta_i"),
            "frame_policy": self.frame_policy,
            "boundary_frames": copy.deepcopy(self.boundary_frames),
            "units": {"length": "m", "angle": "rad"},
            "geometry_tolerances": {"parallel_axis_sine": self.parallel_tolerance},
            "base_link": self.base_link,
            "tip_link": self.tip_link,
            "source_urdf": self.source_urdf,
            "source_sha256": self.source_sha256,
            "input_names": list(self.input_names),
            "input_limits": {name: list(bounds) for name, bounds in self.input_limits.items()},
            "rows": [asdict(row) for row in self.rows],
            "base_transform": self.base_transform.tolist(),
            "tool_transform": self.tool_transform.tolist(),
            "mapping": "q_row = multiplier * q[source_name] + offset; direction is metadata only",
            "warnings": list(self.warnings),
            "validation": self.validation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DHModel:
        """Load exported DH without needing the original URDF or a ROS install."""
        if not isinstance(data, Mapping):
            raise ValueError("A DH document must be an object")
        if (type(data.get("schema_version")) is not int or data.get("schema_version") != 1
                or data.get("convention") not in ("standard", "modified")):
            raise ValueError("Expected schema_version=1 and convention='standard' or 'modified'")
        convention = data["convention"]
        orders = {TRANSFORM_ORDERS[convention]}
        if convention == "modified":
            orders.add("Tx(a) Rx(alpha) Rz(theta) Tz(d)")  # v0.3: commuting X transforms
        order = data.get("transform_order", TRANSFORM_ORDERS[convention])
        if not isinstance(order, str) or order not in orders:
            raise ValueError("transform_order does not match the DH convention")
        indexing = ("a_i, alpha_i, d_i, theta_i" if convention == "standard"
                    else "a_(i-1), alpha_(i-1), d_i, theta_i")
        if data.get("parameter_indexing", indexing) != indexing:
            raise ValueError("parameter_indexing does not match the DH convention")
        if data.get("units") != {"length": "m", "angle": "rad"}:
            raise ValueError("Only metre / radian DH documents are supported")
        required = {"base_link", "tip_link", "input_names", "rows", "base_transform", "tool_transform"}
        if not required.issubset(data):
            raise ValueError(f"Missing DH document fields: {sorted(required - data.keys())}")
        for endpoint in ("base_link", "tip_link"):
            if not isinstance(data[endpoint], str) or not data[endpoint]:
                raise ValueError(f"{endpoint} must be a non-empty string")
        if not isinstance(data["input_names"], (list, tuple)):
            raise ValueError("input_names must be an array")
        names = tuple(data["input_names"])
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("input_names must be unique non-empty strings")
        rows = []
        joint_names = set()
        if not isinstance(data["rows"], (list, tuple)):
            raise ValueError("rows must be an array")
        numeric_fields = ("a", "alpha", "d_offset", "theta_offset", "multiplier", "offset")
        for item in data["rows"]:
            if not isinstance(item, Mapping):
                raise ValueError("Each DH row must be an object")
            try:
                row = DHRow(**item)
            except TypeError as error:
                raise ValueError(f"Invalid DH row fields: {error}") from error
            if (not isinstance(row.joint_type, str)
                    or row.joint_type not in {"revolute", "continuous", "prismatic"}
                    or row.source_name not in names or type(row.direction) is not int
                    or row.direction not in {-1, 1}
                    or not isinstance(row.joint_name, str) or not row.joint_name
                    or row.joint_name in joint_names or not isinstance(row.geometry_case, str)):
                raise ValueError("Invalid DH row joint, input mapping, or traversal direction")
            numbers = [getattr(row, name) for name in numeric_fields]
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in numbers):
                raise ValueError("DH parameters and joint mappings must be real numbers")
            if not np.isfinite(np.asarray(numbers, dtype=float)).all():
                raise ValueError("DH parameters and joint mappings must be finite")
            values = asdict(row)
            values.update((name, float(values[name])) for name in numeric_fields)
            row = DHRow(**values)
            joint_names.add(row.joint_name)
            rows.append(row)
        limits = {}
        stored_limits = data.get("input_limits", {})
        if not isinstance(stored_limits, Mapping):
            raise ValueError("input_limits must be an object")
        for name, bounds in stored_limits.items():
            if name not in names or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("Invalid input_limits")
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, Real))
                   for value in bounds):
                raise ValueError("Joint limits must be numbers or null")
            lower, upper = (None if item is None else float(item) for item in bounds)
            if any(value is not None and not math.isfinite(value) for value in (lower, upper)):
                raise ValueError("Joint limits must be finite or null")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("Joint lower limit exceeds upper limit")
            limits[name] = (lower, upper)
        stored_warnings = data.get("warnings", [])
        if (not isinstance(stored_warnings, (list, tuple))
                or any(not isinstance(warning, str) for warning in stored_warnings)):
            raise ValueError("warnings must be an array of strings")
        if data.get("validation") is not None and not isinstance(data["validation"], Mapping):
            raise ValueError("validation must be an object or null")
        tolerances = data.get("geometry_tolerances", {"parallel_axis_sine": 1e-12})
        if not isinstance(tolerances, Mapping):
            raise ValueError("geometry_tolerances must be an object")
        parallel = tolerances.get("parallel_axis_sine", 1e-12)
        if (isinstance(parallel, bool) or not isinstance(parallel, Real)
                or not math.isfinite(parallel) or not 0 < parallel < 1):
            raise ValueError("parallel_axis_sine must be finite and between 0 and 1")
        base_transform = _checked_transform(data["base_transform"], "base_transform")
        tool_transform = _checked_transform(data["tool_transform"], "tool_transform")
        policy = data.get("frame_policy", "legacy")
        if policy not in ("legacy", "controller"):
            raise ValueError("Unknown frame_policy")
        boundary = data.get("boundary_frames", {})
        if not isinstance(boundary, Mapping):
            raise ValueError("boundary_frames must be an object")
        if policy == "controller" and rows and not boundary:
            raise ValueError("controller frame_policy requires boundary_frames")
        checked_boundary = {}
        if boundary:
            matrix_names = ("mount_transform", "base_alignment", "tool_alignment", "flange_to_tip")
            if set(boundary) != {"mount_link", "flange_link", *matrix_names}:
                raise ValueError("boundary_frames requires mount/flange links and four transforms")
            for key in ("mount_link", "flange_link"):
                if not isinstance(boundary[key], str) or not boundary[key]:
                    raise ValueError(f"boundary_frames.{key} must be a nonempty string")
                checked_boundary[key] = boundary[key]
            for key in matrix_names:
                checked_boundary[key] = _checked_transform(boundary[key], f"boundary_frames.{key}").tolist()
            for product, actual, name in ((np.array(checked_boundary["mount_transform"]) @ checked_boundary["base_alignment"], base_transform, "base_transform"),
                                          (np.array(checked_boundary["tool_alignment"]) @ checked_boundary["flange_to_tip"], tool_transform, "tool_transform")):
                if not np.allclose(product, actual, atol=1e-8, rtol=1e-12):
                    raise ValueError(f"boundary_frames is inconsistent with {name}")
        return cls(
            base_link=data["base_link"], tip_link=data["tip_link"], rows=tuple(rows),
            base_transform=base_transform, tool_transform=tool_transform,
            input_names=names, input_limits=limits, warnings=tuple(stored_warnings),
            validation=data.get("validation"), source_urdf=data.get("source_urdf"),
            source_sha256=data.get("source_sha256"),
            parallel_tolerance=float(parallel),
            convention=convention,
            frame_policy=policy, boundary_frames=checked_boundary,
        )


def _next_frame(previous: np.ndarray, point: np.ndarray, axis: np.ndarray,
                warnings: list[str], label: str,
                parallel_tolerance: float) -> tuple[np.ndarray, str]:
    """Construct F_i with z on the next joint axis, and x a common normal.

    Cross products avoid loss in 1 - dot(u,v)**2 near parallel axes.
    Nonparallel x = unit(u x v) allows signed a and a deterministic alpha.
    """
    origin = previous[:3, 3]
    u = previous[:3, 2]
    v = axis / np.linalg.norm(axis)
    cross = np.cross(u, v)
    sine = float(np.linalg.norm(cross))
    delta = point - origin
    scale = max(np.finfo(float).tiny, float(np.linalg.norm(delta)))
    distance_tolerance = 32.0 * np.finfo(float).eps * scale
    if sine > parallel_tolerance:
        x = cross / sine
        w = np.cross(x, u)
        distance_next = -float(np.dot(delta, w)) / sine
        next_origin = point + distance_next * v
        separation = abs(float(np.dot(delta, x)))
        case = "intersecting" if separation <= 1e-12 * scale else "skew"
        if sine < 1e-6:
            warnings.append(
                f"{label}: nearly parallel axes (sin(angle)={sine:.3g}); "
                "standard DH can have very large offsets and lose numerical precision."
            )
        return frame(next_origin, v, x), case
    if sine > 32.0 * np.finfo(float).eps:
        warnings.append(
            f"{label}: axis sin(angle)={sine:.3g} is within the parallel-axis "
            f"tolerance {parallel_tolerance:.3g}; using the parallel gauge."
        )
    # In the parallel gauge choose zero longitudinal displacement from the
    # preceding DH origin. Keeping v itself preserves antiparallel orientation.
    next_origin = point - float(np.dot(delta, v)) * v
    radial = next_origin - origin
    radial -= float(np.dot(radial, v)) * v
    distance = float(np.linalg.norm(radial))
    x = radial / distance if distance > distance_tolerance else perpendicular(v, previous[:3, 0])
    opposite = float(np.dot(u, v)) < 0.0
    case = ("parallel" if distance > distance_tolerance else "collinear")
    if opposite:
        case = "antiparallel_" + case
    return frame(next_origin, v, x), case


def _path_references(chain: Any) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Physical joint frames at geometric zero; no mimic offsets or hardware I/O."""
    references = []
    current = np.eye(4)
    boundary: dict[str, Any] = {}
    tail = np.eye(4)
    for step in chain.steps:
        before = current.copy()
        origin = step.joint.origin if step.direction == 1 else inverse(step.joint.origin)
        current = current @ origin
        if step.joint.type == "fixed":
            tail = tail @ origin
            continue
        if not references:
            boundary["mount_link"] = step.joint.parent if step.direction == 1 else step.joint.child
            boundary["mount_transform"] = before
        # For reverse edges motion acts in the entering child frame, before
        # inverse(origin). Its signed axis already accounts for reversal.
        references.append(current.copy() if step.direction == 1 else before)
        boundary["flange_link"] = step.joint.child if step.direction == 1 else step.joint.parent
        boundary["flange_pose_zero"] = current.copy()
        tail = np.eye(4)
    if references:
        boundary["flange_to_tip"] = tail
    return references, boundary


def _reference_x(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    for column in (0, 1):
        preferred = reference[:3, column]
        radial = preferred - float(np.dot(preferred, axis)) * axis
        if np.linalg.norm(radial) > 1e-12:
            return radial / np.linalg.norm(radial)
    return perpendicular(axis)


def _controller_frames(axes: Any, references: list[np.ndarray], mount: np.ndarray,
                       warnings: list[str], parallel_tolerance: float) -> tuple[list[np.ndarray], list[str]]:
    """Craig frames close to URDF joint X axes, anchored to mount and flange.

    Intermediate origins lie at the outgoing common-normal foot on their own
    signed joint axis. The last origin is the actual joint origin, independent
    of tip selection. The start origin is the mount origin projected onto the
    first axis; its longitudinal joint displacement therefore stays in row 1.
    """
    frames = []
    cases = []
    for index, ((point, axis), reference) in enumerate(zip(axes, references)):
        preferred = _reference_x(axis, reference)
        if index + 1 == len(axes):
            frames.append(frame(point, axis, preferred))
            continue
        outgoing, case = _next_frame(frame(point, axis, preferred), *axes[index + 1],
                                     warnings, f"joint pair {index + 1}/{index + 2}",
                                     parallel_tolerance)
        x = outgoing[:3, 0].copy()
        score = float(np.dot(x, preferred))
        if score < -1e-14 or (abs(score) <= 1e-14 and np.dot(x, np.cross(axis, preferred)) < 0):
            x = -x
        origin = point + float(np.dot(outgoing[:3, 3] - point, axis)) * axis
        frames.append(frame(origin, axis, x))
        cases.append(case)
    point, axis = axes[0]
    origin = point + float(np.dot(mount[:3, 3] - point, axis)) * axis
    frames.insert(0, frame(origin, axis, _reference_x(axis, mount)))
    return frames, cases


def _controller_theta(angle: float) -> float:
    # Keep near-half-turn offsets positive, like the controller's +180 deg.
    # Adding exactly 2*pi changes only the representative, not the geometry.
    if angle < 0 and math.isclose(angle, -math.pi, rel_tol=0, abs_tol=1e-4):
        return angle + 2 * math.pi
    return 0.0 if angle == 0 else angle


def validate_model(model: DHModel, chain: Any, *, samples: int = 100, seed: int = 0,
                   position_tolerance: float = 1e-7,
                   rotation_tolerance: float = 1e-7) -> dict[str, Any]:
    """Compare DH FK against the independent original-URDF transform chain."""
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 0:
        raise ValueError("samples must be a non-negative integer")
    for name, tolerance in [("position", position_tolerance), ("rotation", rotation_tolerance)]:
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"{name}_tolerance must be finite and positive")
    # A few sampled poses can accidentally hide cancellation of metre-billion
    # DH offsets. Reject a requested tolerance below the representation's
    # geometric floating-point resolution estimate, even when samples happen
    # to pass. This is a conservative eligibility check, not a certified bound.
    lengths = [math.hypot(*model.base_transform[:3, 3]),
               math.hypot(*model.tool_transform[:3, 3])]
    lengths.extend(math.hypot(row.a, row.d_offset) for row in model.rows)
    try:
        position_resolution = float(np.finfo(float).eps * math.fsum(lengths))
    except OverflowError:
        position_resolution = math.inf
    generator = np.random.default_rng(seed)
    ranges = {}
    for name in model.input_names:
        lower, upper = model.input_limits.get(name, (None, None))
        types = {row.joint_type for row in model.rows if row.source_name == name}
        span = 0.25 if types == {"prismatic"} else math.pi
        if lower is None and upper is None:
            lower, upper = -span, span
        elif lower is None:
            lower = upper - 2 * span
        elif upper is None:
            upper = lower + 2 * span
        ranges[name] = (lower, upper)
    max_position = max_rotation = 0.0
    worst_position: dict[str, float] = {}
    worst_rotation: dict[str, float] = {}
    for index in range(samples + 1):
        positions = {name: (0.0 if index == 0 else float(generator.uniform(*ranges[name])))
                     for name in model.input_names}
        dh_pose = model.fk(positions)
        urdf_pose = chain.fk(positions)
        position, rotation = pose_errors(dh_pose, urdf_pose)
        if not np.isfinite(dh_pose).all() or not np.isfinite(urdf_pose).all():
            position = rotation = math.inf
        if position > max_position:
            max_position, worst_position = position, positions.copy()
        if rotation > max_rotation:
            max_rotation, worst_rotation = rotation, positions.copy()
    return {
        "passed": (max_position <= position_tolerance and max_rotation <= rotation_tolerance
                   and position_resolution <= position_tolerance),
        "samples": samples + 1,
        "random_samples": samples,
        "seed": int(seed),
        "max_position_error": max_position,
        "max_rotation_error": max_rotation,
        "position_tolerance": position_tolerance,
        "rotation_tolerance": rotation_tolerance,
        "position_resolution_estimate": position_resolution,
        "resolution_estimate_note": "machine epsilon times total fixed DH/B/C translation magnitude; "
                                    "conservative eligibility estimate, not a certified error bound",
        "worst_position_inputs": worst_position,
        "worst_rotation_inputs": worst_rotation,
        "input_sampling_ranges": {name: list(bounds) for name, bounds in ranges.items()},
        "scope": "URDF model equivalence only; zero inputs plus deterministic random samples; "
                 "mimic-dependent limits are not enforced; no physical calibration claim",
    }


def extract_dh(
    source: str | Path | URDFModel, base_link: str, tip_link: str, *,
    convention: str = "standard",
    validate: bool = True, samples: int = 100, seed: int = 0,
    position_tolerance: float = 1e-7, rotation_tolerance: float = 1e-7,
    parallel_tolerance: float = 1e-12,
) -> DHModel:
    """Extract standard or Craig modified DH with the complete base->tip FK.

    Axes are aligned with signed physical motion, including reverse edges.
    Consequently direction is traversal metadata, never another sign on q.
    """
    convention = normalize_convention(convention)
    if (not math.isfinite(parallel_tolerance) or not 0 < parallel_tolerance < 1):
        raise ValueError("parallel_tolerance must be finite and between 0 and 1")
    if isinstance(source, URDFModel):
        urdf = source
        source_path = None
        source_hash = None
    else:
        path = Path(source)
        urdf = URDFModel.from_file(path)
        source_path = str(path.resolve())
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    chain = urdf.chain(base_link, tip_link)
    axes, tip_zero = chain.zero_geometry()
    warnings: list[str] = []
    frames = []
    cases = []
    boundary: dict[str, Any] = {}
    tool_transform = tip_zero
    if axes:
        references, physical = _path_references(chain)
        frames, cases = _controller_frames(axes, references, physical["mount_transform"],
                                           warnings, parallel_tolerance)
        boundary = {
            "mount_link": physical["mount_link"], "flange_link": physical["flange_link"],
            "mount_transform": physical["mount_transform"].tolist(),
            "base_alignment": (inverse(physical["mount_transform"]) @ frames[0]).tolist(),
            "tool_alignment": (inverse(frames[-1]) @ physical["flange_pose_zero"]).tolist(),
            "flange_to_tip": physical["flange_to_tip"].tolist(),
        }
        tool_transform = np.array(boundary["tool_alignment"]) @ physical["flange_to_tip"]
        if any(not np.allclose(boundary[key], np.eye(4), atol=1e-10, rtol=0)
               for key in ("base_alignment", "tool_alignment")):
            warnings.append("URDF boundary frames need fixed axis alignment; exported boundary_frames "
                            "separates installation Base, DH alignment and flange-to-tip Tool.")
    rows = []
    for index, step in enumerate(chain.active_steps):
        transform = inverse(frames[index]) @ frames[index + 1]
        theta = _controller_theta(math.atan2(-float(transform[0, 1]), float(transform[0, 0])))
        alpha = math.atan2(-float(transform[1, 2]), float(transform[2, 2])) if index else 0.0
        a = float(transform[0, 3]) if index else 0.0
        d = float(-math.sin(alpha) * transform[1, 3] + math.cos(alpha) * transform[2, 3])
        rebuilt = mdh_matrix(theta, d, a, alpha)
        position_error, rotation_error = pose_errors(transform, rebuilt)
        local_scale = max(1.0, float(np.linalg.norm(transform[:3, 3])))
        if (not math.isfinite(position_error) or not math.isfinite(rotation_error)
                or not math.isfinite(local_scale)
                or position_error > 1e-10 * local_scale or rotation_error > 1e-10):
            raise ConversionError(f"Cannot factor joint {step.joint.name!r} into controller MDH: "
                                  f"{position_error:.3g} m, {rotation_error:.3g} rad")
        name, multiplier, offset = chain.joint_mapping(step.joint.name)
        rows.append(DHRow(
            joint_name=step.joint.name, joint_type=step.joint.type,
            a=a, alpha=alpha, d_offset=d, theta_offset=theta,
            source_name=name, multiplier=multiplier, offset=offset,
            direction=step.direction, geometry_case=cases[index - 1] if index else "initial",
        ))
    if any(step.direction < 0 for step in chain.active_steps):
        warnings.append("Reverse edges use signed reversed axes; do not multiply q by direction again.")
    if any(row.source_name != row.joint_name or row.offset != 0.0 or row.multiplier != 1.0 for row in rows):
        warnings.append("Mimic coordinates use the exported affine input mapping; DH rows are not all independent.")
    result = DHModel(
        base_link=base_link, tip_link=tip_link, rows=tuple(rows),
        base_transform=frames[0] if axes else np.eye(4),
        tool_transform=tool_transform,
        input_names=tuple(chain.input_names), input_limits=dict(chain.input_limits),
        warnings=tuple(warnings), source_urdf=source_path, source_sha256=source_hash,
        parallel_tolerance=parallel_tolerance, convention="modified",
        frame_policy="controller", boundary_frames=boundary,
    )
    if convention == "standard":
        # Craig M_i=X_(i-1) Z_i(q), X_0=I. Standard A_i=Z_i(q) X_i.
        # Choose X_n=I: both modes share physical endpoints and the same B/C.
        # Neither mode absorbs the fixed flange-to-tip geometry into its table.
        modified_rows = result.rows
        standard_rows = []
        for index, row in enumerate(modified_rows):
            following = modified_rows[index + 1] if index + 1 < len(modified_rows) else None
            standard_rows.append(replace(
                row, a=following.a if following else 0.0,
                alpha=following.alpha if following else 0.0,
                geometry_case=following.geometry_case if following else "terminal",
            ))
        result.rows = tuple(standard_rows)
        result.convention = "standard"
    if validate:
        result.validation = validate_model(result, chain, samples=samples, seed=seed,
                                           position_tolerance=position_tolerance,
                                           rotation_tolerance=rotation_tolerance)
        if not result.validation["passed"]:
            raise ValidationError(result.validation)
    return result


def extract_standard_dh(
    source: str | Path | URDFModel, base_link: str, tip_link: str, *,
    validate: bool = True, samples: int = 100, seed: int = 0,
    position_tolerance: float = 1e-7, rotation_tolerance: float = 1e-7,
    parallel_tolerance: float = 1e-12,
) -> DHModel:
    """Backward-compatible entry point for standard DH extraction."""
    return extract_dh(source, base_link, tip_link, convention="standard", validate=validate,
                      samples=samples, seed=seed, position_tolerance=position_tolerance,
                      rotation_tolerance=rotation_tolerance, parallel_tolerance=parallel_tolerance)


def extract_modified_dh(
    source: str | Path | URDFModel, base_link: str, tip_link: str, *,
    validate: bool = True, samples: int = 100, seed: int = 0,
    position_tolerance: float = 1e-7, rotation_tolerance: float = 1e-7,
    parallel_tolerance: float = 1e-12,
) -> DHModel:
    """Extract Craig MDH; each row stores a_(i-1), alpha_(i-1), d_i, theta_i."""
    return extract_dh(source, base_link, tip_link, convention="modified", validate=validate,
                      samples=samples, seed=seed, position_tolerance=position_tolerance,
                      rotation_tolerance=rotation_tolerance, parallel_tolerance=parallel_tolerance)
