"""Known-mechanism checks independent of the converter's URDF FK code."""

import json

import numpy as np
import pytest

from urdf_dh import DHModel, extract_standard_dh
from urdf_dh.core import ConversionError

from .helpers import KnownRobot, KnownRow, transform


def assert_pose(actual, expected, atol=1e-7):
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=atol, rtol=0)
    np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=atol, rtol=0)
    np.testing.assert_allclose(actual[3], [0, 0, 0, 1], atol=atol, rtol=0)


def exercise_oracle(tmp_path, robot, *, base="base_link", tip="tip_link", seed=41):
    path = robot.write(tmp_path / "known.urdf")
    model = extract_standard_dh(path, base, tip, samples=30, seed=seed)
    assert model.validation["passed"]
    assert model.validation["samples"] >= 30
    assert model.validation["max_position_error"] < 1e-7
    assert model.validation["max_rotation_error"] < 1e-7
    rng = np.random.default_rng(seed)
    assert_pose(model.fk(), robot.fk(base_link=base, tip_link=tip))
    for _ in range(25):
        # The selected chain accepts only its independent inputs. A mimic's
        # driver may legitimately sit outside that chain.
        values = {name: float(q) for name, q in zip(model.input_names, rng.uniform(-0.7, 0.7, len(model.input_names)))}
        assert_pose(model.fk(values), robot.fk(values, base_link=base, tip_link=tip))
    return model


def test_planar_two_revolute_has_full_arbitrary_frame_equivalence(tmp_path):
    robot = KnownRobot([
        KnownRow("revolute", 0.45, 0.0, 0.12, 0.23),
        KnownRow("continuous", 0.32, 0.0, 0.04, -0.31),
    ])
    model = exercise_oracle(tmp_path, robot)
    assert len(model.rows) == 2
    assert_pose(model.fk([0.4, -0.2]), robot.fk([0.4, -0.2]))


@pytest.mark.parametrize("a,alpha", [
    (0.2, 0.0),                   # parallel, separate axes
    (0.0, 0.0),                   # collinear axes
    (0.2, np.pi),                 # antiparallel, separate axes
    (0.0, np.pi),                 # collinear but opposite axes
    (0.0, -np.pi),                # same degeneracy with the other angle sign
    (0.0, 0.91),                  # intersecting axes
    (0.27, -1.13),                # skew axes
    (-0.27, 0.73),                # signed DH a in the generating mechanism
    (0.2, 1e-7),                  # must retain a small nonzero crossing angle
    (0.2, 1e-9),                  # small angle with finite, well-conditioned offsets
    (0.2, 1e-10),
    (0.2, np.pi - 1e-7),          # almost antiparallel
])
def test_axis_geometry_cases_against_known_mechanism(tmp_path, a, alpha):
    robot = KnownRobot([
        KnownRow("revolute", a, alpha, 0.16, -0.4),
        KnownRow("revolute", 0.24, -0.2, -0.07, 0.32),
    ], seed=14)
    exercise_oracle(tmp_path, robot)


def test_six_axis_arm_with_intersecting_wrist(tmp_path):
    robot = KnownRobot([
        KnownRow("revolute", 0.0, -np.pi / 2, 0.31, 0.2),
        KnownRow("revolute", 0.42, 0.0, 0.0, -0.8),
        KnownRow("revolute", 0.33, 0.0, 0.0, 0.4),
        KnownRow("revolute", 0.0, np.pi / 2, 0.17, -0.2),
        KnownRow("revolute", 0.0, -np.pi / 2, 0.0, 0.7),
        KnownRow("revolute", 0.0, 0.0, 0.09, -0.5),
    ], seed=82)
    model = exercise_oracle(tmp_path, robot)
    assert len(model.rows) == 6


def test_mixed_revolute_prismatic_and_json_roundtrip(tmp_path):
    robot = KnownRobot([
        KnownRow("revolute", 0.21, np.pi / 2, 0.13, 0.32),
        KnownRow("prismatic", 0.0, np.pi, 0.17, -0.4),
        KnownRow("prismatic", -0.12, 0.3, 0.2, 0.1),
        KnownRow("revolute", 0.15, -0.5, -0.04, 0.9),
    ])
    model = exercise_oracle(tmp_path, robot)
    restored = DHModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    q = [0.23, -0.08, 0.16, -0.3]
    assert_pose(restored.fk(q), robot.fk(q))
    assert_pose(restored.fk(), model.fk())


@pytest.mark.parametrize("base,tip,expected_rows", [
    ("link_1", "link_3", 2),
    ("joint_frame_1", "joint_frame_4", 3),
    ("link_3", "link_1", 2),
    ("tip_link", "base_link", 4),
    ("link_2", "joint_frame_2", 0),
    ("link_2", "link_2", 0),
])
def test_forward_and_reverse_subchains(tmp_path, base, tip, expected_rows):
    robot = KnownRobot([
        KnownRow("revolute", 0.2, 0.7, 0.12, 0.3),
        KnownRow("prismatic", 0.13, -0.3, 0.1, -0.4),
        KnownRow("revolute", 0.0, np.pi, 0.18, 0.2),
        KnownRow("continuous", 0.2, -np.pi / 2, 0.07, -0.1),
    ], seed=129)
    model = exercise_oracle(tmp_path, robot, base=base, tip=tip)
    assert len(model.rows) == expected_rows


def test_mimic_rotations_with_multiplier_offset_and_dependency_chain(tmp_path):
    robot = KnownRobot([
        KnownRow("revolute", 0.2, 0.4, 0.1, 0.2),
        KnownRow("revolute", 0.3, np.pi, -0.1, -0.3),
        KnownRow("continuous", 0.1, -0.6, 0.2, 0.4),
    ], mimic={
        "joint_2": ("joint_1", -1.7, 0.23),
        "joint_3": ("joint_2", 0.6, -0.14),
    })
    model = exercise_oracle(tmp_path, robot)
    assert len(model.rows) == 3
    assert_pose(model.fk({"joint_1": 0.7}), robot.fk({"joint_1": 0.7}))
    restored = DHModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert_pose(restored.fk({"joint_1": -0.31}), robot.fk({"joint_1": -0.31}))


def test_mimic_driver_outside_selected_subchain(tmp_path):
    robot = KnownRobot([
        KnownRow("revolute", 0.2, 0.4, 0.1, 0.2),
        KnownRow("revolute", 0.3, np.pi, -0.1, -0.3),
        KnownRow("revolute", 0.1, -0.6, 0.2, 0.4),
    ], mimic={"joint_3": ("joint_1", -0.8, 0.2)})
    exercise_oracle(tmp_path, robot, base="link_2", tip="tip_link")


@pytest.mark.parametrize("seed", range(12))
def test_random_known_dh_chains_in_unrelated_urdf_frames(tmp_path, seed):
    rng = np.random.default_rng(700 + seed)
    rows = []
    for _ in range(int(rng.integers(1, 8))):
        special_angles = [0.0, np.pi, -np.pi, np.pi / 2, -np.pi / 2, rng.uniform(-2.5, 2.5)]
        rows.append(KnownRow(
            "prismatic" if rng.random() < 0.35 else "revolute",
            0.0 if rng.random() < 0.3 else rng.uniform(-0.5, 0.5),
            float(rng.choice(special_angles)),
            rng.uniform(-0.4, 0.4),
            rng.uniform(-2.0, 2.0),
        ))
    robot = KnownRobot(rows, seed=seed,
                       base=transform(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)),
                       tool=transform(rng.uniform(-0.3, 0.3, 3), rng.uniform(-1, 1, 3)))
    model = exercise_oracle(tmp_path, robot, seed=seed)
    assert len(model.rows) == len(rows)


def test_nominal_antiparallel_axes_survive_accumulated_frame_roundoff(tmp_path):
    # Stress case 455: nominal antiparallel axes became sin(angle)=7.4e-15
    # after serial frame changes. Treating that noise as a real crossing angle
    # created 4.3e13 metre DH offsets and centimetre cancellation errors.
    robot = KnownRobot([
        KnownRow("revolute", 0.0, -0.16065278566587082, 0.3198664266789486, 1.7657739385783424),
        KnownRow("prismatic", -0.465427702118529, np.pi / 2, 0.19009789568159374, -0.5991740368718386),
        KnownRow("revolute", -0.19949052372770848, np.pi, 0.02408246498017108, -0.035338168994196284),
        KnownRow("prismatic", -0.39980926890949686, 0.0, -0.2263320995145094, -0.8766345526169466),
        KnownRow("prismatic", -0.25433140865587833, np.pi / 2, 0.12963783849215027, 1.3724627924926547),
        KnownRow("revolute", -0.10060476246844574, np.pi / 2, -0.026202019180463054, 1.6201456620861014),
        KnownRow("revolute", 0.4411797266485312, np.pi, 0.2727236145322308, -0.06753341374452715),
        KnownRow("revolute", -0.1634357977648596, np.pi, -0.11217142316761636, -0.3940585645170702),
    ], seed=455,
        mimic={"joint_8": ("joint_1", -0.8, 0.17)},
        base=transform((0.5944509866896208, -0.3160596189821805, -0.3244307947682994),
                       (-0.06919002817283859, 0.23873060676887725, 0.09129582887954957)),
        tool=transform((-0.1079927993583906, -0.2787222891766042, -0.2602305161486353),
                       (0.7941126520671364, -0.8230158252761006, 0.9956116302397318)),
    )
    exercise_oracle(tmp_path, robot, base="tip_link", tip="base_link", seed=455)


def test_ill_conditioned_near_parallel_geometry_refuses_unverified_result(tmp_path):
    distance = 1e10
    robot = KnownRobot([
        KnownRow("revolute", 0.2, 1e-10, distance, 0.3),
        KnownRow("revolute", 0.2, 0.4, -distance, -0.1),
    ], seed=0)
    # The URDF geometry remains compact. Very remote common-normal footpoints
    # make the equivalent standard DH table intrinsically ill-conditioned.
    shift = transform((0, 0, -distance))
    robot.link_gauges[0] = shift @ robot.link_gauges[0]
    robot.joint_gauges[1] = shift @ robot.joint_gauges[1]
    path = robot.write(tmp_path / "ill_conditioned.urdf")
    with pytest.raises(ConversionError):
        extract_standard_dh(path, "base_link", "tip_link", samples=40)
