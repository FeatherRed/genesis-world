# Intent

Shows how a `KinematicEntity` acts as a visualization-only "ghost" that plays a reference motion without participating in physics.

# API Surface

- `go2_kinematic.py` — Loads two Go2 quadrupeds (`urdf/go2/urdf/go2.urdf`): one physics-simulated entity that falls under gravity, and a second `KinematicEntity` ghost that follows a sinusoidal joint trajectory; `-v` opens the viewer.

# Constraints

- Single-file demo; CPU backend (`gs.init(backend=gs.cpu)`).

# Dependencies

- None beyond the core package.

# See Also

- `../locomotion/` — Go2 used as a real RL-trained controller rather than a ghost reference.
