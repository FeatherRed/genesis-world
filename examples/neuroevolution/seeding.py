"""The single source of truth for the neuroevolution seed rules.

Every trainer in ``examples/`` derives its episode seeds and its evolution seed from the one master seed through this module, so the documented reproduction guarantees hold for every task.
``np.random.SeedSequence(master_seed).spawn(2)`` yields the ``(evolution, env)`` streams, in that order, and the evolution stream comes first.
The evolution stream feeds the TensorNEAT RNG, the env stream feeds :func:`generation_seed_table`.
Every policy of one generation shares the same ordered triple of episode seeds, and every generation gets a fresh triple.
Episodes are paired across the population on purpose: the fitness ranking of a generation is then a paired comparison, not a draw-quality lottery.
The seed table is prefix-stable, so asking for more generations never changes the rows already materialized.
A zero perturbation reproduces the authored fixed reset bit-for-bit, which keeps every unperturbed run of a task identical to its pre-neuroevolution baseline.
Concretely, ``evolution_seed(0) == 3757552657`` and ``generation_seed_table(0, 3)`` is ``[[1454269175, 757617120, 1421746521], [1313752747, 908599966, 53557616], [1683657442, 1375977993, 1654802006]]``.

One sentence per docstring line, numpy plus the stdlib only, no task-specific defaults: the caller owns its action count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EPISODES_PER_POLICY",
    "Perturbation",
    "evolution_seed",
    "generation_seed_table",
    "perturbation_from_seed",
]

#: Episodes every policy is scored on within one generation.
EPISODES_PER_POLICY = 3


@dataclass(frozen=True)
class Perturbation:
    """Reset-time perturbation amplitudes for the env module to sample from.

    Each field is the **maximum absolute offset**; the env module owns the RNG and draws, e.g., ``uniform(-base_pos, +base_pos)`` per axis.
    ``base_pos`` is in metres, ``base_rpy`` in radians, ``joint_pos`` in radians.
    The all-zero default reproduces the original fixed-reset task, which is why ``is_zero`` exists.

    ``perturbed_init_qpos`` deliberately does not live here: RNG semantics belong to the environment, not to a compiled-graph module.
    """

    base_pos: float = 0.0
    base_rpy: float = 0.0
    joint_pos: float = 0.0

    @property
    def is_zero(self) -> bool:
        """Whether every amplitude is zero (the unperturbed reset)."""
        return self.base_pos == 0.0 and self.base_rpy == 0.0 and self.joint_pos == 0.0


def generation_seed_table(master_seed, num_generations, episodes=EPISODES_PER_POLICY):
    """Return the ``(num_generations, episodes)`` int64 table of episode seeds.

    The table is drawn from the env stream of ``SeedSequence(master_seed).spawn(2)`` (the evolution stream is :func:`evolution_seed`), one row at a time: ``rng.choice(2**31, size=episodes, replace=False)`` per row.
    Drawing one row at a time keeps a row's seeds distinct and, unlike one big ``replace=False`` draw, keeps the table **prefix-stable** -- ``table(seed, N)[i]`` is identical for every ``N > i``.

    Args:
        master_seed (int): Seed of the whole run.
        num_generations (int): Number of generation rows to materialize.
        episodes (int): Episodes per policy and row length.

    Returns:
        np.ndarray: ``(num_generations, episodes)`` int64, every row strictly distinct.
    """
    num_generations = int(num_generations)
    episodes = int(episodes)
    if num_generations < 0 or episodes < 1:
        raise ValueError(f"num_generations must be >= 0 and episodes >= 1, got {num_generations} and {episodes}")
    _, env_seq = np.random.SeedSequence(master_seed).spawn(2)
    rng = np.random.default_rng(env_seq)
    table = np.empty((num_generations, episodes), dtype=np.int64)
    for generation in range(num_generations):
        table[generation] = rng.choice(2**31, size=episodes, replace=False)
    return table


def evolution_seed(master_seed):
    """Return the unsigned 32-bit seed of the evolution stream of ``master_seed``.

    This is the sibling of the env stream :func:`generation_seed_table` draws from, so the trainer's evolution RNG is independent of -- and unaffected by -- the episode seeds.

    Args:
        master_seed (int): Seed of the whole run.

    Returns:
        int: Non-negative seed that fits in ``uint32``.
    """
    evolution_seq, _ = np.random.SeedSequence(master_seed).spawn(2)
    return int(evolution_seq.generate_state(1, dtype=np.uint32)[0])


def _uniform(rng, amplitude, size):
    """Return ``size`` offsets uniform in ``[-amplitude, +amplitude]``, consuming ``size`` draws."""
    amplitude = float(amplitude)
    return rng.uniform(-amplitude, amplitude, size=size)


def _rpy_to_quat(rpy):
    """Return the w-first quaternion of ``rpy``, transcribed from genesis.utils.geom._np_xyz_to_quat, rpy=True.

    The transcription is verbatim, including the float64 expression order and the ``sign = 1.0`` of the ``rpy=True`` branch, so the result is bit-identical to ``genesis.utils.geom.xyz_to_quat(rpy, rpy=True)``.
    Genesis is not imported here because this module must stay importable without a simulator install.

    Args:
        rpy (np.ndarray): ``(..., 3)`` roll-pitch-yaw angles in radians, float64.

    Returns:
        np.ndarray: ``(..., 4)`` ``(qw, qx, qy, qz)`` quaternion in the input dtype.
    """
    rpy = np.asarray(rpy, dtype=np.float64)
    out = np.empty((*rpy.shape[:-1], 4), dtype=rpy.dtype)
    rpy2 = 0.5 * rpy
    roll2, pitch2, yaw2 = rpy2[..., 0], rpy2[..., 1], rpy2[..., 2]
    cosr, sinr = np.cos(roll2), np.sin(roll2)
    cosp, sinp = np.cos(pitch2), np.sin(pitch2)
    cosy, siny = np.cos(yaw2), np.sin(yaw2)
    sign = 1.0
    out[..., 0] = cosr * cosp * cosy + sign * sinr * sinp * siny
    out[..., 1] = sinr * cosp * cosy - sign * cosr * sinp * siny
    out[..., 2] = cosr * sinp * cosy + sign * sinr * cosp * siny
    out[..., 3] = cosr * cosp * siny - sign * sinr * sinp * cosy
    return out


def perturbation_from_seed(seed, perturbation, num_dof):
    """Draw one reset perturbation from ``seed`` alone.

    The draw order is fixed (and independent of the amplitudes, so the stream never shifts): first the three ``base_pos`` offsets, then the three ``base_rpy`` angles, then the ``num_dof`` ``joint_pos`` offsets, all uniform per component in ``[-p, +p]``.
    A zero ``Perturbation`` draws nothing and returns exact zeros, with ``base_quat == [1, 0, 0, 0]``.

    ``base_quat`` is the standard w-first roll-pitch-yaw quaternion of ``base_rpy`` (transcribed from genesis.utils.geom._np_xyz_to_quat, rpy=True), i.e. the small base rotation in world axes that :func:`perturbed_init_qpos` applies on top of the authored init orientation (``q_pert * q_init``).

    Args:
        seed (int): Episode seed of the generation's seed table.
        perturbation (Perturbation): Maximum absolute offsets (metres / radians / radians).
        num_dof (int): Length of the joint offset vector, supplied by the caller's action count.

    Returns:
        dict: ``base_pos`` (3,) float32, ``base_quat`` (4,) float32 w-first, ``joint_pos`` (``num_dof``,) float32.
    """
    num_dof = int(num_dof)
    if num_dof < 0:
        raise ValueError(f"num_dof must be >= 0, got {num_dof}")
    if perturbation.is_zero:
        return {
            "base_pos": np.zeros(3, dtype=np.float32),
            "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "joint_pos": np.zeros(num_dof, dtype=np.float32),
        }
    rng = np.random.default_rng(int(seed))
    base_pos = _uniform(rng, perturbation.base_pos, 3)
    base_rpy = _uniform(rng, perturbation.base_rpy, 3)
    joint_pos = _uniform(rng, perturbation.joint_pos, num_dof)
    return {
        "base_pos": base_pos.astype(np.float32),
        "base_quat": _rpy_to_quat(base_rpy).astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
    }
