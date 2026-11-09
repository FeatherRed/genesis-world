"""Shared helpers and numeric oracles for the go2_neat_kernels verification."""

from __future__ import annotations

import numpy as np

import evoxc

from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import BaseProblem

import go2_neat_kernels as K
from go2_evox_kernels import STANDING_FLAT

F32 = np.float32


def f32(x):
    return np.ascontiguousarray(np.asarray(x, dtype=F32))


class DummyProblem(BaseProblem):
    @property
    def input_shape(self):
        return (K.NUM_OBS,)

    @property
    def output_shape(self):
        return (K.NUM_ACTIONS,)

    def evaluate(self, state, keys, act_fn, transformed):
        raise NotImplementedError


def make_pipeline(pop_size, seed, max_nodes=80, max_conns=640):
    genome = DefaultGenome(K.NUM_OBS, K.NUM_ACTIONS, max_nodes=max_nodes, max_conns=max_conns)
    algo = NEAT(genome, pop_size=pop_size, species_size=3)
    pipe = Pipeline(algo, DummyProblem(), seed=seed, backend="numpy")
    state = pipe.setup()
    config = pipe._static_config(state)
    pop = tuple(K._host(x) for x in state.pop)
    leaves = tuple(K._host(x) for x in evoxc.tree_leaves(state.species))
    return {
        "algo": algo,
        "pipe": pipe,
        "state": state,
        "config": config,
        "pop": pop,
        "leaves": leaves,
        "randkey": K._host(state.randkey),
        "generation": K._host(state.generation),
    }


def np_transform_by_quat(v, q):
    """genesis ``_np_transform_by_quat`` semantics, float64, independent reimpl."""
    v = np.asarray(v, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q
    ww, xx, xy, xz = w * w, x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z
    vx, vy, vz = v
    ox = vx * (xx + ww - yy - zz) + vy * (2.0 * xy - 2.0 * wz) + vz * (2.0 * xz + 2.0 * wy)
    oy = vx * (2.0 * wz + 2.0 * xy) + vy * (ww - xx + yy - zz) + vz * (2.0 * yz - 2.0 * wx)
    oz = vx * (2.0 * xz - 2.0 * wy) + vy * (2.0 * wx + 2.0 * yz) + vz * (ww - xx - yy + zz)
    n2 = ww + xx + yy + zz
    return np.array([ox / n2, oy / n2, oz / n2], dtype=np.float64)


def inv_quat(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def oracle_obs(base_ang_vel_world, base_quat, dof_pos, dof_vel, actions, commands):
    out = []
    for i in range(len(base_quat)):
        inv = inv_quat(base_quat[i])
        body_ang = np_transform_by_quat(base_ang_vel_world[i], inv)
        proj_g = np_transform_by_quat(np.array([0.0, 0.0, -1.0]), inv)
        row = np.concatenate(
            [
                body_ang * K.OBS_SCALES["ang_vel"],
                proj_g,
                np.asarray(commands, dtype=np.float64) * K._COMMAND_OBS_SCALE,
                (np.asarray(dof_pos[i], dtype=np.float64) - STANDING_FLAT) * K.OBS_SCALES["dof_pos"],
                np.asarray(dof_vel[i], dtype=np.float64) * K.OBS_SCALES["dof_vel"],
                np.asarray(actions[i], dtype=np.float64),
            ]
        )
        out.append(row)
    return np.stack(out)


def random_quats(rng, n, unit_fraction=0.5):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scales = rng.uniform(0.5, 1.6, size=(n, 1))
    unit = rng.random((n, 1)) < unit_fraction
    scales = np.where(unit, 1.0, scales)
    return f32(q * scales)


def random_state(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= rng.uniform(0.6, 1.5, size=(n, 1))
    return {
        "base_pos": f32(np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(0.1, 0.6, n)], axis=1)),
        "base_quat": f32(q),
        "base_lin_vel_world": f32(rng.uniform(-2.0, 2.0, size=(n, 3))),
        "base_ang_vel_world": f32(rng.uniform(-3.0, 3.0, size=(n, 3))),
        "dof_pos": f32(STANDING_FLAT + rng.uniform(-0.8, 0.8, size=(n, 12))),
        "dof_vel": f32(rng.uniform(-3.0, 3.0, size=(n, 12))),
    }


def quat_from_rpy(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr
    return np.array([qw, qx, qy, qz], dtype=np.float64)
