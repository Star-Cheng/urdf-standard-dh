"""Independent known-DH oracle and URDF generator used by geometry tests.

This module intentionally does not import the package under test.  Each test
robot starts as a known standard-DH mechanism, then expresses its joints and
links in unrelated URDF frames.  The expected FK is evaluated from the original
table, so an error shared by the URDF parser and extractor cannot mask itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class KnownRow:
    kind: str
    a: float
    alpha: float
    d: float
    theta: float


def standard_dh(row: KnownRow, q: float = 0.0) -> np.ndarray:
    """Closed-form Rz(theta) Tz(d) Tx(a) Rx(alpha), independent of production."""
    theta = row.theta + (q if row.kind != "prismatic" else 0.0)
    d = row.d + (q if row.kind == "prismatic" else 0.0)
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(row.alpha), np.sin(row.alpha)
    return np.array([
        [ct, -st * ca, st * sa, row.a * ct],
        [st, ct * ca, -ct * sa, row.a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """URDF Rz(yaw) Ry(pitch) Rx(roll) with a translation."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = xyz
    return result


def _rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(np.cos(pitch)) > 1e-12:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return float(roll), float(pitch), float(yaw)


def _numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


class KnownRobot:
    """A chain with arbitrary URDF frame gauges and independent DH FK."""

    def __init__(
        self,
        rows: Sequence[KnownRow],
        *,
        seed: int = 3,
        base: np.ndarray | None = None,
        tool: np.ndarray | None = None,
        mimic: Mapping[str, tuple[str, float, float]] | None = None,
    ):
        self.rows = list(rows)
        self.base = base if base is not None else transform((0.31, -0.17, 0.28), (0.27, -0.4, 0.63))
        self.tool = tool if tool is not None else transform((-0.08, 0.13, 0.21), (-0.6, 0.2, 0.47))
        self.mimic = dict(mimic or {})
        self.joint_names = [f"joint_{i + 1}" for i in range(len(rows))]
        rng = np.random.default_rng(seed)
        # H_i locates a joint origin on its true z axis, but gives it any
        # orientation; G_i gives the following body any coordinate frame.
        self.joint_gauges = [
            transform((0, 0, rng.uniform(-0.3, 0.3)), rng.uniform(-1.1, 1.1, size=3))
            for _ in rows
        ]
        self.link_gauges = [
            transform(rng.uniform(-0.4, 0.4, size=3), rng.uniform(-1.1, 1.1, size=3))
            for _ in rows
        ]

    def positions(self, positions: Mapping[str, float] | Sequence[float] | None = None) -> dict[str, float]:
        if positions is None:
            result = {}
        elif isinstance(positions, Mapping):
            result = dict(positions)
        else:
            result = dict(zip(self.joint_names, positions))

        def resolve(name: str) -> float:
            if name in self.mimic:
                parent, multiplier, offset = self.mimic[name]
                return multiplier * resolve(parent) + offset
            return float(result.get(name, 0.0))

        return {name: resolve(name) for name in self.joint_names}

    def link_poses(self, positions=None) -> dict[str, np.ndarray]:
        values = self.positions(positions)
        result = {"base_link": np.eye(4), "mount_link": self.base.copy()}
        prefix = self.base.copy()
        for index, (name, row, joint_gauge, link_gauge) in enumerate(
            zip(self.joint_names, self.rows, self.joint_gauges, self.link_gauges), start=1
        ):
            q = values[name]
            motion = standard_dh(KnownRow(row.kind, 0, 0, 0, 0), q)
            result[f"joint_frame_{index}"] = prefix @ motion @ joint_gauge
            prefix = prefix @ standard_dh(row, q)
            result[f"link_{index}"] = prefix @ link_gauge
        result["tip_link"] = prefix @ self.tool
        return result

    def fk(self, positions=None, base_link="base_link", tip_link="tip_link") -> np.ndarray:
        poses = self.link_poses(positions)
        return np.linalg.inv(poses[base_link]) @ poses[tip_link]

    def write(self, path: Path) -> Path:
        root = ET.Element("robot", name="independent_known_dh_robot")
        links: set[str] = set()

        def add_joint(name, kind, parent, child, origin, axis=None):
            for link in (parent, child):
                if link not in links:
                    ET.SubElement(root, "link", name=link)
                    links.add(link)
            joint = ET.SubElement(root, "joint", name=name, type=kind)
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "origin", xyz=_numbers(origin[:3, 3]), rpy=_numbers(_rpy(origin[:3, :3])))
            if axis is not None:
                ET.SubElement(joint, "axis", xyz=_numbers(axis))
                if kind != "continuous":
                    extent = "0.8" if kind == "prismatic" else "3.1"
                    ET.SubElement(joint, "limit", lower=f"-{extent}", upper=extent, effort="10", velocity="2")
            if name in self.mimic:
                driver, multiplier, offset = self.mimic[name]
                ET.SubElement(joint, "mimic", joint=driver, multiplier=str(multiplier), offset=str(offset))

        add_joint("base_mount", "fixed", "base_link", "mount_link", self.base)
        previous_link = "mount_link"
        previous_gauge = np.eye(4)
        for index, (name, row, joint_gauge, link_gauge) in enumerate(
            zip(self.joint_names, self.rows, self.joint_gauges, self.link_gauges), start=1
        ):
            joint_frame = f"joint_frame_{index}"
            link = f"link_{index}"
            add_joint(
                name, row.kind, previous_link, joint_frame,
                np.linalg.inv(previous_gauge) @ joint_gauge,
                joint_gauge[:3, :3].T @ np.array([0, 0, 1]),
            )
            add_joint(
                f"body_frame_{index}", "fixed", joint_frame, link,
                np.linalg.inv(joint_gauge) @ standard_dh(row) @ link_gauge,
            )
            previous_link, previous_gauge = link, link_gauge
        add_joint("tool_mount", "fixed", previous_link, "tip_link", np.linalg.inv(previous_gauge) @ self.tool)
        ET.indent(root, space="  ")
        path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
        return path


def stress_robot(case: int) -> tuple[KnownRobot, str, str, str]:
    """Reproduce one of 500 seeded geometry stress cases without URDF parsing.

    Groups of 100 cover ordinary geometry, 1e-12 m geometry, 1e9 m geometry,
    compact near-parallel geometry with remote common-normal footpoints, and
    reverse mixed R/P chains with mimic coupling. Ordinary groups have 1-12
    joints; the two latter groups require at least two joints.
    """
    if not 0 <= case < 500:
        raise ValueError("Stress case must be in range(500)")
    group = ["ordinary", "tiny_1e-12", "long_1e9", "near_parallel", "reverse_mimic"][case // 100]
    rng = np.random.default_rng(47000 + case)
    scale = {"tiny_1e-12": 1e-12, "long_1e9": 1e9}.get(group, 1.0)
    count = int(rng.integers(2 if group in {"near_parallel", "reverse_mimic"} else 1, 13))
    rows = []
    for _ in range(count):
        rows.append(KnownRow(
            "prismatic" if rng.random() < 0.35 else "revolute",
            0.0 if rng.random() < 0.25 else float(rng.uniform(-0.5, 0.5) * scale),
            float(rng.choice([0.0, np.pi, -np.pi, np.pi / 2, -np.pi / 2, rng.uniform(-3.0, 3.0)])),
            float(rng.uniform(-0.4, 0.4) * scale), float(rng.uniform(-2.0, 2.0)),
        ))
    mimic = {}
    if group == "reverse_mimic":
        rows[-1] = replace(rows[-1], kind=rows[0].kind)
        mimic[f"joint_{count}"] = ("joint_1", -0.8, 0.17)
    if group == "near_parallel":
        angle = [1e-9, 1e-10][case % 2]
        distance = 1.0 / angle
        rows[0] = replace(rows[0], kind="revolute", alpha=angle, d=distance)
        rows[1] = replace(rows[1], kind="revolute", d=-distance)
    robot = KnownRobot(
        rows, seed=case, mimic=mimic,
        base=transform(rng.uniform(-1, 1, 3) * scale, rng.uniform(-1, 1, 3)),
        tool=transform(rng.uniform(-0.3, 0.3, 3) * scale, rng.uniform(-1, 1, 3)),
    )
    for matrices in (robot.joint_gauges, robot.link_gauges):
        for matrix in matrices:
            matrix[:3, 3] *= scale
    if group == "near_parallel":
        shift = transform((0, 0, -distance))
        robot.link_gauges[0] = shift @ robot.link_gauges[0]
        robot.joint_gauges[1] = shift @ robot.joint_gauges[1]
    base, tip = ("tip_link", "base_link") if group == "reverse_mimic" else ("base_link", "tip_link")
    return robot, base, tip, group
