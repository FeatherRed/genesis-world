import sys, time
sys.path.insert(0, "examples/locomotion")
import numpy as np
import go2_neat_kernels as k

print("constants", k.NUM_OBS, k.NUM_ACTIONS, k.EPISODES_PER_POLICY, k.CLIP_ACTIONS, k.TERMINATION_LIMIT_DEG, k.OBS_SCALES)
print("default commands", k.DEFAULT_COMMANDS, k.DEFAULT_COMMANDS.dtype)
print("perturbation", k.Perturbation(), k.Perturbation().is_zero, k.Perturbation(base_pos=0.05).is_zero)

rows = 6
rng = np.random.default_rng(3)
base_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (rows, 1))
base_quat[1] = [0.99, 0.05, -0.02, 0.03]
base_quat[1] /= np.linalg.norm(base_quat[1])

t0 = time.perf_counter()
obs_exe = k.build_obs(rows)
t1 = time.perf_counter()
print("obs compiled in", t1 - t0, "s")
obs = k.run_obs(
    obs_exe,
    base_ang_vel_world=rng.normal(size=(rows, 3)).astype(np.float32),
    base_quat=base_quat,
    dof_pos=rng.normal(size=(rows, 12)).astype(np.float32),
    dof_vel=rng.normal(size=(rows, 12)).astype(np.float32),
    actions=rng.normal(size=(rows, 12)).astype(np.float32),
    commands=k.DEFAULT_COMMANDS,
)
print("obs", obs.shape, obs.dtype)
print(obs[0])

adv = k.build_advance(rows, max_episode_length=1000)
out = k.run_advance(
    adv,
    base_pos=np.tile(np.array([0.0, 0.0, 0.3], np.float32), (rows, 1)),
    base_quat=base_quat,
    base_lin_vel_world=rng.normal(size=(rows, 3)).astype(np.float32),
    base_ang_vel_world=rng.normal(size=(rows, 3)).astype(np.float32),
    dof_pos=rng.normal(size=(rows, 12)).astype(np.float32),
    dof_vel=rng.normal(size=(rows, 12)).astype(np.float32),
    actions=np.full((rows, 12), 500.0, np.float32),
    last_actions=np.zeros((rows, 12), np.float32),
    returns=np.zeros((rows,), np.float32),
    alive=np.ones((rows,), bool),
    episode_length=np.zeros((rows,), np.int32),
    error_mask=np.zeros((rows,), bool),
    commands=k.DEFAULT_COMMANDS,
)
for key, value in out.items():
    print(key, getattr(value, "shape", None), getattr(value, "dtype", None), value if value.ndim == 1 else "")

fit = k.build_fitness(2)
print("fitness", k.run_fitness(fit, np.arange(6, dtype=np.float32)))
print("OK")
