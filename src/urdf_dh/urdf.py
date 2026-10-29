"""URDF tree parsing and forward kinematics without a ROS dependency.

All transforms map coordinates from the child frame into the parent frame.
Angles are in radians and distances are in metres, as required by URDF.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


class URDFError(ValueError):
    """An invalid URDF, selected chain, or joint-position argument."""


@dataclass(frozen=True)
class Mimic:
    joint: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    origin: np.ndarray
    axis: np.ndarray
    parent: str
    child: str
    lower: float | None = None
    upper: float | None = None
    mimic: Mimic | None = None


@dataclass(frozen=True)
class Step:
    joint: Joint
    direction: int


_TYPES = frozenset(("fixed", "revolute", "continuous", "prismatic", "planar", "floating"))
_ANGULAR = frozenset(("revolute", "continuous"))
_SUPPORTED = frozenset(("fixed", "revolute", "continuous", "prismatic"))


def _number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise URDFError(f"{context} must be a finite number; got {value!r}") from exc
    if not math.isfinite(result):
        raise URDFError(f"{context} must be finite; got {value!r}")
    return result


def _vector(value: str | None, default: tuple[float, float, float], context: str) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    parts = value.split()
    if len(parts) != 3:
        raise URDFError(f"{context} must contain exactly three numbers")
    return np.array([_number(part, context) for part in parts], dtype=float)


def _name(element: ET.Element, attribute: str, context: str) -> str:
    result = element.get(attribute, "").strip()
    if not result:
        raise URDFError(f"{context} requires a nonempty {attribute!r} attribute")
    return result


def _one(element: ET.Element, tag: str, context: str, *, required: bool = False) -> ET.Element | None:
    found = element.findall(tag)
    if len(found) > 1:
        raise URDFError(f"{context} has multiple <{tag}> elements")
    if not found:
        if required:
            raise URDFError(f"{context} requires a <{tag}> element")
        return None
    return found[0]


def _origin(element: ET.Element | None, context: str) -> np.ndarray:
    result = np.eye(4)
    if element is None:
        return result
    xyz = _vector(element.get("xyz"), (0.0, 0.0, 0.0), f"{context} origin xyz")
    roll, pitch, yaw = _vector(element.get("rpy"), (0.0, 0.0, 0.0), f"{context} origin rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF uses fixed-axis roll/pitch/yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    result[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    result[:3, 3] = xyz
    return result


def _inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _motion(joint: Joint, value: float) -> np.ndarray:
    result = np.eye(4)
    if joint.type in _ANGULAR:
        x, y, z = joint.axis
        cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        sine, cosine = math.sin(value), math.cos(value)
        result[:3, :3] = cosine * np.eye(3) + (1.0 - cosine) * np.outer(joint.axis, joint.axis) + sine * cross
    elif joint.type == "prismatic":
        result[:3, 3] = joint.axis * value
    return result


class URDFModel:
    """A validated URDF link tree; meshes, inertias, and transmissions are ignored."""

    def __init__(self, links: Sequence[str], joints: Mapping[str, Joint]):
        self.links = tuple(links)
        self.joints = dict(joints)
        self._parents: dict[str, Joint] = {}
        self._mappings: dict[str, tuple[str, float, float]] = {}
        self._validate_tree()
        self._validate_mimics()

    @classmethod
    def from_file(cls, path: str | Path) -> URDFModel:
        """Read expanded URDF XML. Xacro must be expanded by the caller."""
        try:
            xml = Path(path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise URDFError(f"Cannot read URDF file {str(path)!r}: {exc}") from exc
        return cls.from_xml(xml)

    @classmethod
    def from_xml(cls, xml: str | bytes) -> URDFModel:
        if not isinstance(xml, (str, bytes)):
            raise URDFError("URDF XML must be text or bytes")
        declaration = br"<!\s*(?:DOCTYPE|ENTITY)\b" if isinstance(xml, bytes) else r"<!\s*(?:DOCTYPE|ENTITY)\b"
        if re.search(declaration, xml, re.IGNORECASE):
            raise URDFError("DOCTYPE and ENTITY declarations are not supported in URDF XML")
        try:
            root = ET.fromstring(xml)
        except (ET.ParseError, TypeError, ValueError) as exc:
            raise URDFError(f"Invalid URDF XML: {exc}") from exc
        for element in root.iter():
            if "xacro" in element.tag or any(
                marker in value
                for value in (*element.attrib.values(), element.text or "")
                for marker in ("${", "$(")
            ):
                raise URDFError("Unexpanded xacro detected; expand xacro to URDF before parsing")
        if root.tag != "robot":
            raise URDFError("URDF root element must be <robot>")
        links: list[str] = []
        names: set[str] = set()
        for element in root.findall("link"):
            name = _name(element, "name", "link")
            if name in names:
                raise URDFError(f"Duplicate link name {name!r}")
            names.add(name)
            links.append(name)
        joints: dict[str, Joint] = {}
        for element in root.findall("joint"):
            name = _name(element, "name", "joint")
            context = f"joint {name!r}"
            if name in joints:
                raise URDFError(f"Duplicate joint name {name!r}")
            kind = _name(element, "type", context)
            if kind not in _TYPES:
                raise URDFError(f"{context} has unknown type {kind!r}")
            parent = _one(element, "parent", context, required=True)
            child = _one(element, "child", context, required=True)
            assert parent is not None and child is not None
            origin = _origin(_one(element, "origin", context), context)
            axis_element = _one(element, "axis", context)
            axis = _vector(axis_element.get("xyz") if axis_element is not None else None,
                           (1.0, 0.0, 0.0), f"{context} axis")
            if kind in ("fixed", "floating"):
                # These joint types do not use an axis; CAD exporters may emit
                # a zero placeholder. Still parse it above to catch bad numbers.
                axis = np.array((1.0, 0.0, 0.0))
            else:
                # Scaling first avoids overflow for large but finite vectors.
                scale = float(np.max(np.abs(axis)))
                if scale == 0.0:
                    raise URDFError(f"{context} axis must be nonzero")
                axis = axis / scale
                axis /= np.linalg.norm(axis)
            limit = _one(element, "limit", context)
            lower = upper = None
            if limit is not None:
                if limit.get("lower") is not None:
                    lower = _number(limit.get("lower"), f"{context} lower limit")
                if limit.get("upper") is not None:
                    upper = _number(limit.get("upper"), f"{context} upper limit")
                if lower is not None and upper is not None and lower > upper:
                    raise URDFError(f"{context} lower limit is greater than upper limit")
                for attribute in ("effort", "velocity"):
                    if limit.get(attribute) is not None:
                        _number(limit.get(attribute), f"{context} {attribute} limit")
            mimic_element = _one(element, "mimic", context)
            mimic = None
            if mimic_element is not None:
                mimic = Mimic(
                    _name(mimic_element, "joint", f"{context} mimic"),
                    _number(mimic_element.get("multiplier", "1"), f"{context} mimic multiplier"),
                    _number(mimic_element.get("offset", "0"), f"{context} mimic offset"),
                )
            joints[name] = Joint(name, kind, origin, axis,
                                 _name(parent, "link", f"{context} parent"),
                                 _name(child, "link", f"{context} child"),
                                 lower, upper, mimic)
        return cls(links, joints)

    def _validate_tree(self) -> None:
        if not self.links:
            raise URDFError("URDF must contain at least one link")
        if len(self.links) != len(set(self.links)):
            raise URDFError("URDF contains duplicate link names")
        links = set(self.links)
        children: dict[str, list[str]] = {name: [] for name in self.links}
        for joint in self.joints.values():
            for role, link in (("parent", joint.parent), ("child", joint.child)):
                if link not in links:
                    raise URDFError(f"Joint {joint.name!r} references unknown {role} link {link!r}")
            if joint.child in self._parents:
                raise URDFError(f"Link {joint.child!r} has multiple parent joints")
            self._parents[joint.child] = joint
            children[joint.parent].append(joint.child)
        roots = links - self._parents.keys()
        if len(roots) != 1:
            raise URDFError("URDF topology must be one connected tree with exactly one root link; "
                            f"found {len(roots)} roots (a cycle or disconnected links may be present)")
        self.root_link = next(iter(roots))
        seen: set[str] = set()
        pending = [self.root_link]
        while pending:
            link = pending.pop()
            if link in seen:
                raise URDFError("URDF topology contains a cycle")
            seen.add(link)
            pending.extend(children[link])
        if seen != links:
            raise URDFError("URDF topology contains a disconnected component or cycle")

    def _validate_mimics(self) -> None:
        # Explicit stacks also support chains longer than Python's recursion limit.
        for initial in self.joints:
            if self.joints[initial].type == "fixed":
                if self.joints[initial].mimic is not None:
                    raise URDFError(f"Fixed joint {initial!r} cannot mimic another joint")
                continue
            if initial in self._mappings:
                continue
            path: list[str] = []
            visiting: set[str] = set()
            name = initial
            while name not in self._mappings:
                if name in visiting:
                    raise URDFError(f"Mimic cycle detected at joint {name!r}")
                visiting.add(name)
                joint = self.joints[name]
                path.append(name)
                if joint.mimic is None:
                    self._mappings[name] = (name, 1.0, 0.0)
                    break
                if joint.type not in _ANGULAR and joint.type != "prismatic":
                    raise URDFError(f"Joint {name!r} of type {joint.type!r} cannot mimic")
                source = self.joints.get(joint.mimic.joint)
                if source is None:
                    raise URDFError(f"Joint {name!r} mimics unknown joint {joint.mimic.joint!r}")
                same_kind = (joint.type in _ANGULAR and source.type in _ANGULAR) or (
                    joint.type == source.type == "prismatic"
                )
                if not same_kind:
                    raise URDFError(f"Mimic joint {name!r} and source {source.name!r} must have "
                                    "matching rotational/prismatic types")
                name = source.name
            for name in reversed(path):
                joint = self.joints[name]
                if joint.mimic is None:
                    continue
                source, multiplier, offset = self._mappings[joint.mimic.joint]
                multiplier *= joint.mimic.multiplier
                offset = joint.mimic.multiplier * offset + joint.mimic.offset
                if not math.isfinite(multiplier) or not math.isfinite(offset):
                    raise URDFError(f"Flattened mimic mapping for joint {name!r} is nonfinite")
                self._mappings[name] = (source, multiplier, offset)

    def chain(self, base_link: str, tip_link: str) -> URDFChain:
        """Select the unique tree path, allowing ancestors, descendants, and branches."""
        for role, name in (("base", base_link), ("tip", tip_link)):
            if name not in self.links:
                raise URDFError(f"Unknown {role} link {name!r}")
        ancestors = {base_link}
        cursor = base_link
        while cursor in self._parents:
            cursor = self._parents[cursor].parent
            ancestors.add(cursor)
        downward: list[Step] = []
        cursor = tip_link
        while cursor not in ancestors:
            joint = self._parents[cursor]
            downward.append(Step(joint, 1))
            cursor = joint.parent
        common = cursor
        steps: list[Step] = []
        cursor = base_link
        while cursor != common:
            joint = self._parents[cursor]
            steps.append(Step(joint, -1))
            cursor = joint.parent
        steps.extend(reversed(downward))
        for step in steps:
            if step.joint.type not in _SUPPORTED:
                raise URDFError(f"Selected chain includes unsupported {step.joint.type!r} joint "
                                f"{step.joint.name!r}; only fixed, revolute, continuous, and "
                                "prismatic joints are supported")
        return URDFChain(self, base_link, tip_link, tuple(steps))


class URDFChain:
    """A selected path with affine mimic mappings and direct URDF FK."""

    def __init__(self, model: URDFModel, base_link: str, tip_link: str, steps: tuple[Step, ...]):
        self.model = model
        self.base_link = base_link
        self.tip_link = tip_link
        self.steps = steps
        self.active_steps = tuple(step for step in steps if step.joint.type != "fixed")
        self.input_names = tuple(dict.fromkeys(self.joint_mapping(step.joint.name)[0]
                                              for step in self.active_steps))
        self.input_limits = {name: (model.joints[name].lower, model.joints[name].upper)
                             for name in self.input_names}

    def joint_mapping(self, name: str) -> tuple[str, float, float]:
        """Return (independent source, multiplier, offset) for a moving joint.

        The mapping is the joint's physical URDF coordinate, without reversing
        signs for a backwards path. ``zero_geometry`` handles that direction.
        """
        try:
            return self.model._mappings[name]
        except KeyError as exc:
            raise URDFError(f"Unknown or fixed joint {name!r} has no position mapping") from exc

    def _inputs(self, positions: Mapping[str, float] | Sequence[float] | None) -> dict[str, float]:
        if positions is None:
            return dict.fromkeys(self.input_names, 0.0)
        if isinstance(positions, Mapping):
            unknown = set(positions) - set(self.input_names)
            if unknown:
                raise URDFError(f"Unknown independent joint position(s): {sorted(map(str, unknown))}; "
                                f"expected {self.input_names}")
            return {name: _number(positions.get(name, 0.0), f"position for joint {name!r}")
                    for name in self.input_names}
        if isinstance(positions, (str, bytes)):
            raise URDFError("Positions must be a mapping or a one-dimensional numeric sequence")
        try:
            values = np.asarray(positions, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise URDFError("Positions must be a mapping or a one-dimensional numeric sequence") from exc
        if values.ndim != 1 or len(values) != len(self.input_names):
            raise URDFError(f"Positions must contain exactly {len(self.input_names)} values in order "
                            f"{self.input_names}")
        return {name: _number(value, f"position for joint {name!r}")
                for name, value in zip(self.input_names, values)}

    def resolve_positions(self, positions: Mapping[str, float] | Sequence[float] | None = None) -> dict[str, float]:
        """Resolve input coordinates into each moving joint's actual coordinate."""
        inputs = self._inputs(positions)
        result: dict[str, float] = {}
        for step in self.active_steps:
            source, multiplier, offset = self.joint_mapping(step.joint.name)
            result[step.joint.name] = _number(multiplier * inputs[source] + offset,
                                              f"resolved position for joint {step.joint.name!r}")
        return result

    def fk(self, positions: Mapping[str, float] | Sequence[float] | None = None) -> np.ndarray:
        """Compute base-to-tip FK directly from origin and joint motion matrices."""
        resolved = self.resolve_positions(positions)
        result = np.eye(4)
        for step in self.steps:
            joint = step.joint
            transform = joint.origin @ _motion(joint, resolved.get(joint.name, 0.0))
            result = result @ (transform if step.direction == 1 else _inverse(transform))
        if not np.isfinite(result).all():
            raise URDFError("Forward kinematics overflowed; check geometry and joint positions")
        return result

    def zero_geometry(self) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Return moving-axis lines and tip pose with every physical joint at zero.

        This geometric reference deliberately ignores mimic offsets. The axis
        direction is the derivative for positive *physical* URDF joint motion,
        including the sign needed when the selected path traverses a joint back.
        """
        axes: list[tuple[np.ndarray, np.ndarray]] = []
        result = np.eye(4)
        for step in self.steps:
            joint = step.joint
            if step.direction == 1:
                result = result @ joint.origin
                if joint.type != "fixed":
                    axes.append((result[:3, 3].copy(), result[:3, :3] @ joint.axis))
            else:
                if joint.type != "fixed":
                    axes.append((result[:3, 3].copy(), result[:3, :3] @ -joint.axis))
                result = result @ _inverse(joint.origin)
        if not np.isfinite(result).all() or any(not np.isfinite(point).all() for point, _ in axes):
            raise URDFError("Zero geometry overflowed; check URDF origin values")
        return axes, result
