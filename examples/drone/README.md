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
A pre-programmed drone flight simulation that follows a predefined trajectory stored in `fly_traj.pkl`.

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

#### 3.3 Torch-free GPU-resident training

`hover_neat_train.py --torch-free` runs the EvoX Compiler hover trainer end to end with no Torch in the
process, plus `hover_neat_check.py --torch-free` for a cheap environment-level check. Run them from the
repository root:

```bash
PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py --torch-free --pop-size 8 --generations 2 -o out/hover_neat_pop8
PYTHONPATH=/tmp/notorch:$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_train.py --torch-free --pop-size 8 --generations 2 -o out/hover_neat_pop8
PYTHONPATH=$PWD /root/evox-robotics/.venv/bin/python examples/drone/hover_neat_check.py --torch-free --pop-size 2 --horizon 5
```

`--torch-free` implies the zero-copy device residency, defaults the graphs and the evolution to `iree@rocm:0` (an explicit `--evoxc-device` wins), keeps `gs.gpu` in array mode (`performance_mode=False`), and refuses a viewer, a numpy rollout/tell backend or an XLA tell; what stays host-side is only the per-step command draw, the solver-error latch, the `end_code` readback while stopping on done, and the per-generation reset write.

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

## Controls Implementation

The interactive drone uses differential RPM control:
- Forward/Backward movement: Adjusts front/back rotor pairs
- Left/Right movement: Adjusts left/right rotor pairs
- All movements maintain a stable hover while creating directional thrust
- RPM changes are automatically clipped to safe ranges (0-25000 RPM)