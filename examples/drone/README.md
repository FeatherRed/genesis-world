# Drone Examples

This directory contains examples of drone simulations using the Genesis framework.

## Available Examples

### 1. Interactive Drone (`interactive_drone.py`)
A real-time interactive drone simulation where you can control the drone using keyboard inputs:
- Up: Move Forward (North)
- Down: Move Backward (South)
- Left: Move Left (West)
- Right: Move Right (East)
- Space: Increase Thrust (Accelerate)
- Left Shift: Decrease Thrust (Decelerate)
- ESC: Quit

Run with:
```bash
python interactive_drone.py
```

### 2. Automated Flight (`fly.py`)
A pre-programmed drone flight simulation. The rotor-thrust trajectory is hardcoded inline in the script;
there is no `fly_traj.pkl` in the repository.

Run with:
```bash
python fly.py -v
```

### 3. Hover Environment (`hover_env.py`, `hover_train.py`, `hover_eval.py`)

The hover environment (`hover_env.py`) is designed to train a drone to maintain a stable hover position by reaching randomly generated target points. The environment includes:

 - Initialization of the scene and entities (plane, drone and target).
 - Reward functions to provide feedback to the agent based on its performance in reaching the target points.
 - **Command resampling to generate new random target points** and environment reset functionalities to ensure continuous training.

**Acknowledgement**: The reward design is inspired by [Champion-level drone racing using deep
reinforcement learning (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4.pdf)

#### 3.0 Installation

At this stage, we have defined the environments. Now, we use the PPO implementation from `rsl-rl` to train the policy. First, install all Python dependencies via `pip`:

```bash
pip install tensorboard "rsl-rl-lib>=5.0.0"
```

`rsl-rl-lib` is not declared in `pyproject.toml`.

#### 3.1 Training

Train the drone hovering policy using the `HoverEnv` environment.

Run with:

```bash
python hover_train.py -e drone-hovering -b 8192 --max-iterations 301
```

Train with visualization:

```bash
python hover_train.py -e drone-hovering -b 8192 --max-iterations 301 -v
```

#### 3.2 Evaluation

Evaluate the trained drone hovering policy.

Run with:

```bash
python hover_eval.py -e drone-hovering --ckpt 300 --record
```

**Note**: If you experience slow performance or encounter other issues
during evaluation, try removing the `--record` option.

For the latest updates, detailed documentation, and additional resources, visit this repository: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv).

#### 3.3 Torch-free GPU-resident training (EvoX Compiler)

`hover_neat_train.py` and `hover_neat_check.py` run the same hover task on the shared EvoX Compiler /
TensorNEAT layer in `examples/neuroevolution`. `--torch-free` runs the whole loop with `torch` unimportable.
Both `evoxc` and `tensorneat` (0.2.0, from its `evoxc` branch) are optional and not declared in
`pyproject.toml`.

Run from the repository root; `PYTHONPATH` must contain the checkout, otherwise an installed `genesis`
shadows it and device residency dies on the missing `genesis.utils.device_buffer`:

```bash
PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py --torch-free --pop-size 8 --generations 2 -o out/hover_neat_pop8
PYTHONPATH=/tmp/notorch:$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py --torch-free --pop-size 8 --generations 2
PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --torch-free --pop-size 2 --horizon 5
```

`--torch-free` implies `--residency device`, initializes `gs.init(backend=gs.gpu, performance_mode=False)`,
defaults the graphs and the evolution to `iree@rocm:0` (an explicit `--evoxc-device` wins) and refuses a
viewer, a numpy rollout/tell backend or an XLA tell. Every run prints `torch_in_process=False`, and the
check CLI ends with `torch_free 1`. `performance_mode=True` also runs Torch-free but is not bit-identical
(same population, `returns` up to 1.12e-08 apart over 30 steps, `fitness` and `seeds` exact), so the
deterministic ndarray mode stays on.

Device residency keeps every graph input and output on `rocm:0` (0 off-device leaves across initialization,
policy forward, per-step advance, rewards, fitness, the tell and the next generation's population); the
per-step host work left is the command RNG draw, one solver-error mask copy and the `end_code` readback
while `--stop-when-done` is on. The numpy residency requires torch: `DroneEntity.set_propellers_rpm` is
unusable without it (numpy input raises `AttributeError: 'ScalarNdarray' object has no attribute 'T'` at
`drone_entity.py:84`, a DLPack input raises `TypeError` at `torch_free.py:643`).

Notable flags of `hover_neat_train.py`: `--pop-size` (8), `--generations` (2), `--horizon`, `--master-seed`,
`--perturb-base-pos`/`--perturb-base-rpy` (0.05), `--no-perturbation`, `-e`, `-o` (`out/hover_neat`), `-g`,
`--evoxc-backend`/`--evoxc-device`, `--tell-backend {numpy,iree,xla}`, `--pjrt-plugin`,
`--residency {numpy,device}`, `--stop-when-done`, `--torch-free`, `-v/--vis`, `--quiet`;
`hover_neat_check.py` adds `--parity`, `--residency-parity`, `--imports`, `--perturb` and defaults to
`--pop-size 4 --generations 1 --horizon 20`.

Measured throughput, torch blocked, full episodes, shared MI300X VF / ROCm 7.1 (wall times vary ±10-15 %):
setup and compile 8.8-9.8 s (~7.9 s of it IREE compiling the tell graph); pop 8 (24 rows) generation 0
160 steps in ~1.5 s (108 steps/s, 2595 env-steps/s), generation 1 58 steps in ~0.4 s (142 steps/s); pop 32
(96 rows) generation 0 ~1.7 s (92 steps/s, 8820 env-steps/s), generation 1 149 steps ~1.2 s (120 steps/s,
11551 env-steps/s) — 90-142 steps/s, env-steps scaling with rows.

Parity: one population scored by both residencies agrees exactly (`returns`, `episode_returns`, `fitness`
all 0.0 on `iree@cpu:0` and `iree@rocm:0`); each residency scoring its own population differs by 2.5e-04 in
generation-0 `returns` (19 of 24 rows <= 1e-6) and 5.4e-07 in generation 1, because IREE device
initialization reproduces TensorNEAT's `Pipeline.setup` float leaves only to 2.384e-07. In the active
`tests/test_hover_device_parity.py` 2 of 5 tests pass; the 3 per-step/reference cases need
`LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1.18.70100` in-process and fail on test-side issues (an `evoxc`
wrapper signature read, a default horizon, a 1e-5 tolerance the measured 2.536e-04 difference exceeds).
pytest is not installed in the project venv: `PYTHONPATH=$PWD:/root/.local/share/pytest-target pytest
tests/test_hover_device_parity.py`.

### 4. Quadcopter Controller (`quadcopter_controller.py` & `fly_route.py`)
A PID-based controller to provide stable point-to-point flight without controlling each rotor. Parameters
have been tuned for controlled flight but not optimized. Stored in `fly_route.py`. Run `fly_route.py` to
test.

Run with:

```bash
python fly_route.py
```

## Technical Details

- The drone model used is the Crazyflie 2.X (`urdf/drones/cf2x.urdf`)
- Base hover RPM is approximately 14468
- Movement is achieved by varying individual rotor RPMs to create directional thrust
- The simulation uses realistic physics including gravity and aerodynamics
- `hover_env.py`, `hover_train.py` and `hover_eval.py` are the PPO baseline and stay on torch + `tensordict`; the EvoX Compiler stack is validated against `hover_legacy_reference.py` instead of importing them

## Controls Implementation

The interactive drone uses differential RPM control:
- Forward/Backward movement: Adjusts front/back rotor pairs
- Left/Right movement: Adjusts left/right rotor pairs
- All movements maintain a stable hover while creating directional thrust
- RPM changes are automatically clipped to safe ranges (0-25000 RPM)
