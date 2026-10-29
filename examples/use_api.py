"""Run from an installed package: python examples/use_api.py."""

import json
from pathlib import Path

from urdf_dh import DHModel, extract_standard_dh


source = Path(__file__).with_name("mixed_arm.urdf")
model = extract_standard_dh(source, "base_link", "tip_link", samples=100, seed=0)
print("Independent inputs:", model.input_names)
print("DH parameters (m, rad):")
for row in model.rows:
    print(row.joint_name, row.a, row.alpha, row.d_offset, row.theta_offset)

# JSON can be stored and loaded later, with no dependency on the original URDF.
loaded = DHModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
print("base->tip pose:\n", loaded.fk({name: 0.1 for name in loaded.input_names}))
print("Validation:", model.validation)
