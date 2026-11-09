import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_xyz
gs.EPS = float(np.finfo(np.float32).eps)

import go2_neat_kernels as K
import vlib as V

F32 = np.float32
rng = np.random.default_rng(4)
n = 3000
pitch = rng.uniform(-8.0, 8.0, n)
roll = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(-1.0, 1.0, n))
yaw = rng.uniform(-180, 180, n)
quats = V.f32(np.stack([V.quat_from_rpy(roll[i], pitch[i], yaw[i]) for i in range(n)]))
exe = K.build_advance(n, max_episode_length=1000)
out = K.run_advance(
    exe,
    base_pos=V.f32(np.tile([0.0, 0.0, 0.3], (n, 1))),
    base_quat=quats,
    base_lin_vel_world=V.f32(np.zeros((n, 3))),
    base_ang_vel_world=V.f32(np.zeros((n, 3))),
    dof_pos=V.f32(np.tile(np.zeros(12, dtype=F32), (n, 1))),
    dof_vel=V.f32(np.zeros((n, 12))),
    actions=V.f32(np.zeros((n, 12))),
    last_actions=V.f32(np.zeros((n, 12))),
    returns=np.zeros(n, dtype=F32),
    alive=np.ones(n, dtype=bool),
    episode_length=np.zeros(n, dtype=np.int32),
    error_mask=np.zeros(n, dtype=bool),
    commands=K.DEFAULT_COMMANDS,
)
rpy = quat_to_xyz(quats, rpy=True, degrees=True)
expected = (np.abs(rpy[:, 0]) > 10.0) | (np.abs(rpy[:, 1]) > 10.0)
got = out["done"]
bad = np.flatnonzero(got != expected)
print(f"mismatches {len(bad)}/{n}")
print("false positives (got True, exp False):", int(np.count_nonzero(got & ~expected)))
print("false negatives (got False, exp True):", int(np.count_nonzero(~got & expected)))
if len(bad):
    i = bad[0]
    print(f"first bad: rpy={rpy[i].tolist()} got={bool(got[i])} exp={bool(expected[i])}")
    worst = np.abs(rpy[bad] - np.sign(rpy[bad]) * 10.0)
    print(f"closest bad |rpy|-10: {np.min(np.abs(np.abs(rpy[bad, :2]) - 10.0)):.4f} deg")
