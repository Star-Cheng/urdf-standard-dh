"""Small SE(3) primitives for standard and Craig modified Denavit-Hartenberg."""

from __future__ import annotations

import math

import numpy as np


TRANSFORM_ORDERS = {
    "standard": "Rz(theta) Tz(d) Tx(a) Rx(alpha)",
    "modified": "Rx(alpha) Tx(a) Rz(theta) Tz(d)",
}


def normalize_convention(convention: str) -> str:
    if not isinstance(convention, str) or convention not in {"standard", "modified", "mdh"}:
        raise ValueError("convention must be 'standard', 'modified' or 'mdh' (Craig)")
    return "modified" if convention == "mdh" else convention


def inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha), angles in radians."""
    if not all(math.isfinite(value) for value in (theta, d, a, alpha)):
        raise ValueError("DH angles and distances must be finite")
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def mdh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Craig/controller: Rx(alpha_(i-1)) Tx(a_(i-1)) Rz(theta_i) Tz(d_i)."""
    if not all(math.isfinite(value) for value in (theta, d, a, alpha)):
        raise ValueError("MDH angles and distances must be finite")
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def pose_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Translation norm (m), SO(3) geodesic error (rad), stable near zero."""
    rotation = actual[:3, :3].T @ expected[:3, :3]
    sine = 0.5 * np.linalg.norm([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])), math.atan2(sine, cosine)


def perpendicular(z: np.ndarray, preferred: np.ndarray | None = None) -> np.ndarray:
    candidates = [] if preferred is None else [preferred]
    candidates.extend(np.eye(3)[i] for i in np.argsort(np.abs(z)))
    for candidate in candidates:
        x = np.asarray(candidate) - np.dot(candidate, z) * z
        length = np.linalg.norm(x)
        if length > 1e-12:
            return x / length
    raise ValueError("Cannot choose a perpendicular direction for the joint axis")


def frame(origin: np.ndarray, z: np.ndarray, x: np.ndarray) -> np.ndarray:
    if not all(np.isfinite(value).all() for value in (origin, z, x)):
        raise ValueError("Joint geometry produced a non-finite coordinate frame")
    z = z / np.linalg.norm(z)
    x = perpendicular(z, x)
    y = np.cross(z, x)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, y, z))
    result[:3, 3] = origin
    return result
