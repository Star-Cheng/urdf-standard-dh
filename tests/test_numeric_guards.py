"""Scale-aware length gauges, finite FK and explicit extraction tolerances."""

import numpy as np
import pytest

from urdf_dh import DHModel, dh_matrix, extract_standard_dh
from .helpers import KnownRobot, KnownRow


def test_small_geometry_keeps_relative_accuracy(tmp_path):
    robot = KnownRobot([
        KnownRow("revolute", .2, 0., .1, -.3),
        KnownRow("revolute", .3, np.pi, .2, .1),
        KnownRow("revolute", .1, .5, -.1, .7),
    ])
    for matrix in [robot.base, robot.tool, *robot.joint_gauges, *robot.link_gauges]:
        matrix[:3, 3] *= 1e-12
    robot.rows = [KnownRow(row.kind, row.a * 1e-12, row.alpha,
                          row.d * 1e-12, row.theta) for row in robot.rows]
    path = robot.write(tmp_path / "tiny.urdf")
    model = extract_standard_dh(path, "base_link", "tip_link", position_tolerance=1e-24)
    for q in ([.3, .2, -.4], [-1.4, .3, 1.2]):
        np.testing.assert_allclose(model.fk(q)[:3, 3], robot.fk(q)[:3, 3], atol=1e-24, rtol=0)


@pytest.mark.parametrize("tolerance", [0, -1e-12, float("nan"), float("inf"), 1.0])
def test_reject_invalid_parallel_tolerance(tmp_path, tolerance):
    with pytest.raises(ValueError, match="parallel_tolerance"):
        extract_standard_dh(tmp_path / "irrelevant.urdf", "base", "tip", parallel_tolerance=tolerance)


def test_nonfinite_dh_parameter_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        dh_matrix(0, float("inf"), 0, 0)


def test_dh_fk_rejects_accumulation_overflow():
    # Two individually finite prismatic displacements can overflow on addition.
    from urdf_dh import DHRow
    rows = tuple(DHRow(f"j{i}", "prismatic", 0., 0., 0., 0., f"j{i}") for i in range(2))
    model = DHModel("base", "tip", rows, np.eye(4), np.eye(4), ("j0", "j1"))
    with pytest.raises(ValueError, match="overflow"):
        model.fk([1e308, 1e308])
