import math
import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_xyz
gs.EPS = float(np.finfo(np.float32).eps)

import go2_neat_kernels as K
import vlib as V

F32 = np.float32
cases = [
    (9.95, 8.0, 0.0),
    (-9.95, -8.0, 30.0),
    (9.7, 8.0, 0.0),
    (0.0, 0.0, 0.0),
    (5.0, 9.95, 0.0),
    (0.0, 8.0, 0.0),
]
quats = V.f32(np.stack([V.quat_from_rpy(*c) for c in cases]))
n = len(quats)
exe = K.build_advance(n, max_episode_length=1000)
out = K.run_advance(
    exe,
    base_pos=V.f32(np.tile([0.0, 0.0, 0.3], (n, 1))),
    base_quat=quats,
    base_lin_vel_world=V.f32(np.zeros((n, 3))),
    base_ang_vel_world=V.f32(np.zeros((n, 3))),
    dof_pos=V.f32(np.tile(np.asarray([0.0] * 12, dtype=F32), (n, 1))),
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
for i, c in enumerate(cases):
    print(f"rpy_in={c} rpy_rec={np.round(rpy[i], 5).tolist()} got={bool(out['done'][i])} exp={bool(expected[i])}")
