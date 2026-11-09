"""Verification harness for examples/locomotion/go2_neat_kernels.py."""

from __future__ import annotations

import math
import time
import traceback

import numpy as np

import evoxc

import go2_neat_kernels as K
from go2_evox_kernels import DT, STANDING_FLAT
import go2_legacy_reference as LR
import genesis as gs
from genesis.utils.geom import quat_to_xyz

# gs.init() sets EPS from np.finfo(np.float32).eps on a float32 backend and the
# numba oracle in geom.py reads it, so mirror it here without a simulator.
gs.EPS = float(np.finfo(np.float32).eps)

import vlib as V

F32 = np.float32
f32 = V.f32
RESULTS = []


def run_check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail))
        print(f"[PASS] {name}: {detail}", flush=True)
        return True
    except Exception as exc:
        RESULTS.append((name, "FAIL", repr(exc)))
        print(f"[FAIL] {name}: {exc!r}", flush=True)
        traceback.print_exc()
        return False


# --- Check 1: obs parity ------------------------------------------------------------

RNG = np.random.default_rng(20261109)


def check_obs():
    rows = 64
    exe = K.build_obs(rows)
    base_ang_vel_world = f32(RNG.uniform(-3.0, 3.0, size=(rows, 3)))
    base_quat = V.random_quats(RNG, rows, unit_fraction=0.5)
    dof_pos = f32(STANDING_FLAT + RNG.uniform(-0.5, 0.5, size=(rows, 12)))
    dof_vel = f32(RNG.uniform(-2.0, 2.0, size=(rows, 12)))
    actions = f32(RNG.uniform(-1.0, 1.0, size=(rows, 12)))
    commands = K.DEFAULT_COMMANDS
    got = K.run_obs(
        exe,
        base_ang_vel_world=base_ang_vel_world,
        base_quat=base_quat,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        actions=actions,
        commands=commands,
    )
    expected = V.oracle_obs(base_ang_vel_world, base_quat, dof_pos, dof_vel, actions, commands)
    diff = np.abs(got.astype(np.float64) - expected)
    worst = float(diff.max())
    assert got.shape == (rows, K.NUM_OBS), got.shape
    assert got.dtype == F32, got.dtype
    assert worst <= 1e-6, f"worst obs diff {worst:.3e} > 1e-6"
    return f"worst abs diff {worst:.3e}"


# --- Check 2: reward parity ---------------------------------------------------------


def advance_reward(exe, state, actions, last_actions, rows, max_episode_length=1000):
    return K.run_advance(
        exe,
        base_pos=state["base_pos"],
        base_quat=state["base_quat"],
        base_lin_vel_world=state["base_lin_vel_world"],
        base_ang_vel_world=state["base_ang_vel_world"],
        dof_pos=state["dof_pos"],
        dof_vel=state["dof_vel"],
        actions=actions,
        last_actions=last_actions,
        returns=np.zeros(rows, dtype=F32),
        alive=np.ones(rows, dtype=bool),
        episode_length=np.zeros(rows, dtype=np.int32),
        error_mask=np.zeros(rows, dtype=bool),
        commands=K.DEFAULT_COMMANDS,
    )


def legacy_totals(state, actions, last_actions):
    n = len(state["base_quat"])
    totals = []
    for i in range(n):
        clipped = np.clip(actions[i], -K.CLIP_ACTIONS, K.CLIP_ACTIONS)
        action_change = last_actions[i].astype(np.float64) - clipped.astype(np.float64)
        terms = LR.walking_reward_terms(
            state["base_pos"][i],
            state["base_quat"][i],
            state["base_lin_vel_world"][i],
            state["base_ang_vel_world"][i],
            state["dof_pos"][i],
            action_change,
        )
        totals.append(float(terms["total"]))
    return np.asarray(totals, dtype=np.float64)


def legacy_total_one(st):
    return float(
        LR.walking_reward_terms(
            st["base_pos"][0],
            st["base_quat"][0],
            st["base_lin_vel_world"][0],
            st["base_ang_vel_world"][0],
            st["dof_pos"][0],
            np.zeros(12),
        )["total"]
    )


def check_reward():
    rows = 24
    exe = K.build_advance(rows, max_episode_length=1000)
    state = V.random_state(rows, seed=11)
    rng = np.random.default_rng(12)
    details = []

    actions_in = f32(rng.uniform(-1.0, 1.0, size=(rows, 12)))
    last_actions = f32(rng.uniform(-1.0, 1.0, size=(rows, 12)))
    got = advance_reward(exe, state, actions_in, last_actions, rows)["reward"].astype(np.float64)
    exp = legacy_totals(state, actions_in, last_actions)
    worst_in = float(np.abs(got - exp).max())
    details.append(f"in-range worst {worst_in:.3e}")
    assert worst_in <= 1e-5, worst_in

    actions_out = f32(rng.uniform(-500.0, 500.0, size=(rows, 12)))
    got_out = advance_reward(exe, state, actions_out, last_actions, rows)["reward"].astype(np.float64)
    exp_out = legacy_totals(state, actions_out, last_actions)
    worst_out = float(np.abs(got_out - exp_out).max())
    details.append(f"clipped worst {worst_out:.3e}")
    assert worst_out <= 1e-5, worst_out

    base = {
        "base_pos": f32([[0.0, 0.0, 0.3]]),
        "base_quat": f32([[1.0, 0.0, 0.0, 0.0]]),
        "base_lin_vel_world": f32([[0.5, 0.0, 0.0]]),
        "base_ang_vel_world": f32([[0.0, 0.0, 0.0]]),
        "dof_pos": f32([STANDING_FLAT]),
        "dof_vel": f32(np.zeros((1, 12))),
    }
    zero = f32(np.zeros((1, 12)))
    adv1 = K.build_advance(1, max_episode_length=1000)

    def reward_of(st, actions=zero, last=zero):
        return float(advance_reward(adv1, st, actions, last, 1)["reward"][0])

    base_reward = reward_of(base)
    expected_base = DT * (1.0 + 0.2 * 1.0)
    assert abs(base_reward - expected_base) <= 1e-6, (base_reward, expected_base)
    legacy_base = legacy_total_one(base)
    assert abs(base_reward - legacy_base) <= 1e-6, (base_reward, legacy_base)

    def delta_of(name, mutate, action=None, last=None):
        st = {key: value.copy() for key, value in base.items()}
        mutate(st)
        act = zero if action is None else action
        lst = zero if last is None else last
        got_delta = reward_of(st, act, lst) - base_reward
        le = LR.walking_reward_terms(
            st["base_pos"][0], st["base_quat"][0], st["base_lin_vel_world"][0],
            st["base_ang_vel_world"][0], st["dof_pos"][0],
            (lst[0].astype(np.float64) - np.clip(act[0], -K.CLIP_ACTIONS, K.CLIP_ACTIONS).astype(np.float64)),
        )
        legacy_delta = float(le["total"]) - legacy_base
        err = abs(got_delta - legacy_delta)
        assert err <= 1e-5, f"{name}: got {got_delta:.6e} legacy {legacy_delta:.6e} err {err:.3e}"
        return got_delta, legacy_delta, err, le

    def m_track_lin(st):
        st["base_lin_vel_world"][0, 0] = 0.8

    def m_track_ang(st):
        st["base_ang_vel_world"][0, 2] = 0.3

    def m_lin_z(st):
        st["base_lin_vel_world"][0, 2] = 0.4

    def m_height(st):
        st["base_pos"][0, 2] = 0.35

    pose_delta = np.zeros((1, 12), dtype=F32)
    pose_delta[0, [0, 5, 9]] = [0.2, -0.15, 0.1]

    def m_pose(st):
        st["dof_pos"][0] = STANDING_FLAT + pose_delta[0]

    deltas = {}
    deltas["tracking_lin_vel"] = delta_of("tracking_lin_vel", m_track_lin)
    deltas["tracking_ang_vel"] = delta_of("tracking_ang_vel", m_track_ang)
    deltas["lin_vel_z"] = delta_of("lin_vel_z", m_lin_z)
    deltas["base_height"] = delta_of("base_height", m_height)
    act = f32(np.zeros((1, 12)))
    act[0, [1, 4, 7]] = [0.1, -0.2, 0.3]
    deltas["action_rate"] = delta_of("action_rate", lambda st: None, action=act)
    deltas["similar_to_default"] = delta_of("similar_to_default", m_pose)

    worst_terms = 0.0
    for name, (gd, ld, err, le) in deltas.items():
        term_value = float(le[name])
        worst_terms = max(worst_terms, err)
        details.append(f"{name}: d={gd:.6e} (legacy term {term_value:.6e}, err {err:.2e})")
    assert worst_terms <= 1e-5, worst_terms
    return "; ".join(details)

# --- Check 3: termination parity ----------------------------------------------------


def build_termination_quats(rng):
    quats = []
    n = 4000
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= rng.uniform(0.5, 1.5, size=(n, 1))
    quats.append(q)
    # roll straddles the 10 deg boundary while pitch stays below it, so a
    # too-strict roll predicate shows up as a false positive.
    n = 3000
    roll = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(-1.0, 1.0, n))
    pitch = rng.uniform(-8.0, 8.0, n)
    yaw = rng.uniform(-180.0, 180.0, n)
    quats.append(np.stack([V.quat_from_rpy(roll[i], pitch[i], yaw[i]) for i in range(n)]))
    n = 2500
    pitch = np.sign(rng.uniform(-1, 1, n)) * (10.0 + rng.uniform(-1.0, 1.0, n))
    roll = rng.uniform(-8.0, 8.0, n)
    yaw = rng.uniform(-180.0, 180.0, n)
    quats.append(np.stack([V.quat_from_rpy(roll[i], pitch[i], yaw[i]) for i in range(n)]))
    n = 500
    pitch = np.sign(rng.uniform(-1, 1, n)) * (90.0 - rng.uniform(0.0, 1e-3, n))
    roll = rng.uniform(-180.0, 180.0, n)
    yaw = rng.uniform(-180.0, 180.0, n)
    quats.append(np.stack([V.quat_from_rpy(roll[i], pitch[i], yaw[i]) for i in range(n)]))
    allq = np.concatenate(quats, axis=0).astype(F32)
    scales = np.where(rng.random((len(allq), 1)) < 0.5, rng.uniform(0.5, 1.5, size=(len(allq), 1)), 1.0)
    return f32(allq * scales)


def check_termination():
    rng = np.random.default_rng(99)
    quats = build_termination_quats(rng)
    n = len(quats)
    chunk = 2000
    exe = K.build_advance(chunk, max_episode_length=1000)
    got = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk):
        block = quats[start : start + chunk]
        m = len(block)
        pad = np.concatenate(
            [block, np.tile(np.array([[1.0, 0, 0, 0]], dtype=F32), (chunk - m, 1))], axis=0
        )
        out = K.run_advance(
            exe,
            base_pos=f32(np.tile([0.0, 0.0, 0.3], (chunk, 1))),
            base_quat=pad,
            base_lin_vel_world=f32(np.zeros((chunk, 3))),
            base_ang_vel_world=f32(np.zeros((chunk, 3))),
            dof_pos=f32(np.tile(STANDING_FLAT, (chunk, 1))),
            dof_vel=f32(np.zeros((chunk, 12))),
            actions=f32(np.zeros((chunk, 12))),
            last_actions=f32(np.zeros((chunk, 12))),
            returns=np.zeros(chunk, dtype=F32),
            alive=np.ones(chunk, dtype=bool),
            episode_length=np.zeros(chunk, dtype=np.int32),
            error_mask=np.zeros(chunk, dtype=bool),
            commands=K.DEFAULT_COMMANDS,
        )
        got[start : start + m] = out["done"][:m]
    rpy = quat_to_xyz(quats, rpy=True, degrees=True)
    expected = (np.abs(rpy[:, 0]) > 10.0) | (np.abs(rpy[:, 1]) > 10.0)
    mismatch = int(np.count_nonzero(got != expected))
    margins = np.minimum(np.abs(np.abs(rpy[:, 0]) - 10.0), np.abs(np.abs(rpy[:, 1]) - 10.0))
    worst = float(margins.min())
    detail = f"{n} quats, {mismatch} mismatches, closest-to-boundary margin {worst:.3e} deg"
    if mismatch:
        bad = np.flatnonzero(got != expected)[:10]
        detail += (
            f"; first bad idx {bad.tolist()} rpy={rpy[bad].round(4).tolist()}"
            f" got={got[bad].tolist()} exp={expected[bad].tolist()}"
        )
    assert mismatch == 0, detail
    return detail


# --- Check 4: first-done / return / mask semantics ----------------------------------


def check_semantics():
    rows = 3
    max_len = 6
    exe = K.build_advance(rows, max_episode_length=max_len)
    n_steps = 6

    base_pos = f32(np.tile([0.0, 0.0, 0.3], (rows, 1)))
    base_lin = f32(np.tile([0.5, 0.0, 0.0], (rows, 1)))
    base_ang = f32(np.zeros((rows, 3)))
    dof_pos = f32(np.tile(STANDING_FLAT, (rows, 1)))
    dof_vel = f32(np.zeros((rows, 12)))
    zeros12 = f32(np.zeros((rows, 12)))

    returns = np.zeros(rows, dtype=F32)
    alive = np.ones(rows, dtype=bool)
    ep_len = np.zeros(rows, dtype=np.int32)

    exp_returns = np.zeros(rows, dtype=np.float64)
    exp_alive = np.ones(rows, dtype=bool)
    exp_ep_len = np.zeros(rows, dtype=np.int32)
    done_count = np.zeros(rows, dtype=int)
    alive_history = []
    done_history = []
    ep_history = []
    returns_history = []
    half = math.radians(45.0) / 2.0
    tilt_quat = f32([math.cos(half), math.sin(half), 0.0, 0.0])

    for t in range(n_steps):
        q = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=F32), (rows, 1))
        if t >= 3:
            q[1] = tilt_quat
        err = np.zeros(rows, dtype=bool)
        if t == 1:
            err[2] = True
        out = K.run_advance(
            exe,
            base_pos=base_pos,
            base_quat=q,
            base_lin_vel_world=base_lin,
            base_ang_vel_world=base_ang,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            actions=zeros12,
            last_actions=zeros12,
            returns=returns,
            alive=alive,
            episode_length=ep_len,
            error_mask=err,
            commands=K.DEFAULT_COMMANDS,
        )
        reward = out["reward"].astype(np.float64)
        exp_returns += reward * exp_alive
        ep_next = exp_ep_len + 1
        died = (ep_next > max_len) | err
        if t >= 3:
            died = died.copy()
            died[1] = True
        exp_done = exp_alive & died
        exp_alive = exp_alive & ~died
        exp_ep_len = np.where(exp_done, 0, ep_next).astype(np.int32)
        done_count += exp_done

        got_done = out["done"]
        np.testing.assert_array_equal(got_done, exp_done, err_msg=f"done mismatch at t={t}")
        np.testing.assert_array_equal(out["alive"], exp_alive, err_msg=f"alive mismatch at t={t}")
        np.testing.assert_array_equal(out["episode_length"], exp_ep_len, err_msg=f"ep_len mismatch at t={t}")
        assert np.allclose(out["returns"], exp_returns, atol=1e-6), (t, out["returns"], exp_returns)
        returns = out["returns"]
        alive = out["alive"]
        ep_len = out["episode_length"]
        alive_history.append(alive.copy())
        done_history.append(got_done.copy())
        ep_history.append(ep_len.copy())
        returns_history.append(returns.copy())

    assert alive_history[-1][0], alive_history[-1]
    assert done_count[0] == 0, done_count
    assert done_count[1] == 1 and done_history[3][1], done_count
    assert not alive_history[3][1] and not alive_history[5][1]
    assert ep_history[3][1] == 0 and ep_history[4][1] == 1 and ep_history[5][1] == 2, ep_history
    assert done_count[2] == 1 and done_history[1][2], done_count
    assert ep_history[1][2] == 0 and ep_history[5][2] == 4, ep_history

    assert returns_history[3][1] > returns_history[2][1], returns_history
    np.testing.assert_allclose(returns_history[5][1], returns_history[3][1], atol=0)
    np.testing.assert_allclose(returns_history[5][2], returns_history[1][2], atol=0)
    assert returns_history[5][0] > returns_history[0][0] > 0
    diffs = np.diff(np.concatenate([[0.0], np.array([float(r[0]) for r in returns_history])]))
    assert abs(diffs.sum() - float(returns_history[5][0])) <= 1e-5
    return f"done_count={done_count.tolist()}, rowA return={float(returns_history[5][0]):.6f}"

# --- Check 5: fitness ---------------------------------------------------------------


def check_fitness():
    npol = 8
    exe = K.build_fitness(npol)
    rng = np.random.default_rng(5)
    returns = f32(rng.uniform(-2.0, 2.0, size=npol * K.EPISODES_PER_POLICY))
    got = K.run_fitness(exe, returns)
    expected = returns.astype(np.float64).reshape(npol, K.EPISODES_PER_POLICY).mean(1)
    err = float(np.abs(got.astype(np.float64) - expected).max())
    assert err <= 1e-6, err
    assert got.shape == (npol,) and got.dtype == F32

    extreme = returns.copy()
    extreme[3] = F32(1e9)
    got2 = K.run_fitness(exe, extreme)
    exp2 = extreme.astype(np.float64).reshape(npol, K.EPISODES_PER_POLICY).mean(1)
    rel = float(np.max(np.abs(got2.astype(np.float64) - exp2) / np.maximum(1.0, np.abs(exp2))))
    assert rel <= 1e-6, rel
    return f"random err {err:.3e}, extreme rel err {rel:.3e}"


# --- Check 6: two generations on the same executables -------------------------------


def build_execs(cache, algo, config, pop, leaves, rows, npol, max_len=1000):
    dev = "cpu:0"
    obs = cache.get(("obs", rows, "numpy", dev), lambda: K.build_obs(rows))
    policy = cache.get(("policy", rows, "numpy", dev), lambda: K.build_policy(algo, config, pop, rows))
    advance = cache.get(
        ("advance", rows, max_len, "numpy", dev),
        lambda: K.build_advance(rows, max_episode_length=max_len),
    )
    fitness = cache.get(
        ("fitness", npol, K.EPISODES_PER_POLICY, "numpy", dev),
        lambda: K.build_fitness(npol),
    )
    tell = cache.get(("tell", npol, "numpy", dev), lambda: K.build_tell(algo, config, pop, leaves, npol))
    return {"obs": obs, "policy": policy, "advance": advance, "fitness": fitness, "tell": tell}


def rollout(execs, pop, rows, rng, n_steps=3):
    obs = f32(rng.uniform(-1.0, 1.0, size=(rows, K.NUM_OBS)))
    returns = np.zeros(rows, dtype=F32)
    alive = np.ones(rows, dtype=bool)
    ep_len = np.zeros(rows, dtype=np.int32)
    last_actions = np.zeros((rows, 12), dtype=F32)
    for _ in range(n_steps):
        actions = K.run_policy(execs["policy"], pop, obs)
        assert actions.shape == (rows, K.NUM_ACTIONS), actions.shape
        state = V.random_state(rows, seed=int(rng.integers(1 << 30)))
        out = K.run_advance(
            execs["advance"],
            base_pos=state["base_pos"],
            base_quat=state["base_quat"],
            base_lin_vel_world=state["base_lin_vel_world"],
            base_ang_vel_world=state["base_ang_vel_world"],
            dof_pos=state["dof_pos"],
            dof_vel=state["dof_vel"],
            actions=actions,
            last_actions=last_actions,
            returns=returns,
            alive=alive,
            episode_length=ep_len,
            error_mask=np.zeros(rows, dtype=bool),
            commands=K.DEFAULT_COMMANDS,
        )
        obs = out["obs"]
        returns = out["returns"]
        alive = out["alive"]
        ep_len = out["episode_length"]
        last_actions = f32(np.clip(actions, -K.CLIP_ACTIONS, K.CLIP_ACTIONS))
    return returns


def check_generations(config):
    npol = 8
    rows = npol * K.EPISODES_PER_POLICY
    algo = config["algo"]
    cache = K.ExecutableCache()
    pop = config["pop"]
    leaves = config["leaves"]
    rng = np.random.default_rng(77)

    hits0, misses0 = cache.hits, cache.misses
    execs1 = build_execs(cache, algo, config["config"], pop, leaves, rows, npol)
    misses_after_build = cache.misses
    hits_after_build = cache.hits
    assert misses_after_build == 5, misses_after_build

    returns1 = rollout(execs1, pop, rows, rng)
    fitness1 = K.run_fitness(execs1["fitness"], returns1)
    assert fitness1.shape == (npol,)
    gen_in = config["generation"]
    tell1 = K.run_tell(
        execs1["tell"],
        randkey=config["randkey"],
        pop_arrays=pop,
        generation=gen_in,
        fitness=fitness1,
        species_leaves=leaves,
    )
    for arr, shape in zip(tell1["pop"], execs1["tell"].pop_shapes):
        assert arr.shape == shape, (arr.shape, shape)

    pop2 = tuple(tell1["pop"])
    assert any(not np.array_equal(pop2[i], pop[i]) for i in range(4)), "tell did not change the population"
    assert pop2[0].shape == pop[0].shape
    leaves2 = tuple(evoxc.tree_leaves(tell1["species"]))
    assert len(leaves2) == len(leaves)

    hits1, misses1 = cache.hits, cache.misses
    execs2 = build_execs(cache, algo, config["config"], pop2, leaves2, rows, npol)
    assert cache.misses == misses1, (cache.misses, misses1)
    assert cache.hits > hits1, (cache.hits, hits1)
    for key in execs1:
        assert execs2[key] is execs1[key], f"{key} executable was rebuilt"

    returns2 = rollout(execs2, pop2, rows, rng)
    fitness2 = K.run_fitness(execs2["fitness"], returns2)
    tell2 = K.run_tell(
        execs2["tell"],
        randkey=tell1["randkey"],
        pop_arrays=pop2,
        generation=tell1["generation"],
        fitness=fitness2,
        species_leaves=leaves2,
    )
    gen_values = [float(gen_in), float(tell1["generation"]), float(tell2["generation"])]
    assert gen_values[1] > gen_values[0] and gen_values[2] > gen_values[1], gen_values
    return (
        f"cache hits {hits0}->{hits_after_build}->{cache.hits}, "
        f"misses {misses0}->{misses_after_build}->{cache.misses}; "
        f"generations {gen_values}; fitness1[:3]={fitness1[:3].tolist()}"
    )

# --- Check 7: timings ---------------------------------------------------------------


def time_build(fn):
    t0 = time.perf_counter()
    obj = fn()
    return time.perf_counter() - t0, obj


def time_calls(fn, warm=3, iters=50):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def check_timings(config, config16):
    npol, rows = 8, 24
    npol16, rows16 = 16, 48
    rng = np.random.default_rng(3)
    lines = []

    t_obs, obs24 = time_build(lambda: K.build_obs(rows))
    t_pol, pol24 = time_build(lambda: K.build_policy(config["algo"], config["config"], config["pop"], rows))
    t_adv, adv24 = time_build(lambda: K.build_advance(rows, max_episode_length=1000))
    t_fit, fit8 = time_build(lambda: K.build_fitness(npol))
    t_tell, tell8 = time_build(
        lambda: K.build_tell(config["algo"], config["config"], config["pop"], config["leaves"], npol)
    )
    lines.append(
        f"compile(rows=24/NP=8 ms): obs {t_obs*1e3:.1f} policy {t_pol*1e3:.1f} advance {t_adv*1e3:.1f} "
        f"fitness {t_fit*1e3:.1f} tell {t_tell*1e3:.1f}"
    )

    obs_in = f32(rng.uniform(-1, 1, size=(rows, K.NUM_OBS)))
    st = V.random_state(rows, seed=4)
    acts = f32(rng.uniform(-1, 1, size=(rows, 12)))
    last = f32(rng.uniform(-1, 1, size=(rows, 12)))
    rets = f32(rng.uniform(-1, 1, size=rows))
    call_obs24 = time_calls(
        lambda: K.run_obs(
            obs24,
            base_ang_vel_world=st["base_ang_vel_world"],
            base_quat=st["base_quat"],
            dof_pos=st["dof_pos"],
            dof_vel=st["dof_vel"],
            actions=acts,
            commands=K.DEFAULT_COMMANDS,
        )
    )
    call_pol24 = time_calls(lambda: K.run_policy(pol24, config["pop"], obs_in))
    call_adv24 = time_calls(
        lambda: K.run_advance(
            adv24,
            base_pos=st["base_pos"],
            base_quat=st["base_quat"],
            base_lin_vel_world=st["base_lin_vel_world"],
            base_ang_vel_world=st["base_ang_vel_world"],
            dof_pos=st["dof_pos"],
            dof_vel=st["dof_vel"],
            actions=acts,
            last_actions=last,
            returns=rets,
            alive=np.ones(rows, dtype=bool),
            episode_length=np.zeros(rows, dtype=np.int32),
            error_mask=np.zeros(rows, dtype=bool),
            commands=K.DEFAULT_COMMANDS,
        )
    )
    call_fit8 = time_calls(lambda: K.run_fitness(fit8, rets))
    lines.append(
        f"warm/step(rows=24/NP=8 ms): obs {call_obs24*1e3:.3f} policy {call_pol24*1e3:.3f} "
        f"advance {call_adv24*1e3:.3f} fitness {call_fit8*1e3:.3f}"
    )

    t_obs2, obs48 = time_build(lambda: K.build_obs(rows16))
    t_pol2, pol48 = time_build(
        lambda: K.build_policy(config16["algo"], config16["config"], config16["pop"], rows16)
    )
    t_adv2, adv48 = time_build(lambda: K.build_advance(rows16, max_episode_length=1000))
    t_fit2, fit16 = time_build(lambda: K.build_fitness(npol16))
    lines.append(
        f"compile(rows=48/NP=16 ms): obs {t_obs2*1e3:.1f} policy {t_pol2*1e3:.1f} "
        f"advance {t_adv2*1e3:.1f} fitness {t_fit2*1e3:.1f}"
    )

    obs_in2 = f32(rng.uniform(-1, 1, size=(rows16, K.NUM_OBS)))
    st2 = V.random_state(rows16, seed=6)
    acts2 = f32(rng.uniform(-1, 1, size=(rows16, 12)))
    last2 = f32(rng.uniform(-1, 1, size=(rows16, 12)))
    rets2 = f32(rng.uniform(-1, 1, size=rows16))
    call_obs48 = time_calls(
        lambda: K.run_obs(
            obs48,
            base_ang_vel_world=st2["base_ang_vel_world"],
            base_quat=st2["base_quat"],
            dof_pos=st2["dof_pos"],
            dof_vel=st2["dof_vel"],
            actions=acts2,
            commands=K.DEFAULT_COMMANDS,
        )
    )
    call_pol48 = time_calls(lambda: K.run_policy(pol48, config16["pop"], obs_in2))
    call_adv48 = time_calls(
        lambda: K.run_advance(
            adv48,
            base_pos=st2["base_pos"],
            base_quat=st2["base_quat"],
            base_lin_vel_world=st2["base_lin_vel_world"],
            base_ang_vel_world=st2["base_ang_vel_world"],
            dof_pos=st2["dof_pos"],
            dof_vel=st2["dof_vel"],
            actions=acts2,
            last_actions=last2,
            returns=rets2,
            alive=np.ones(rows16, dtype=bool),
            episode_length=np.zeros(rows16, dtype=np.int32),
            error_mask=np.zeros(rows16, dtype=bool),
            commands=K.DEFAULT_COMMANDS,
        )
    )
    call_fit16 = time_calls(lambda: K.run_fitness(fit16, rets2))
    lines.append(
        f"warm/step(rows=48/NP=16 ms): obs {call_obs48*1e3:.3f} policy {call_pol48*1e3:.3f} "
        f"advance {call_adv48*1e3:.3f} fitness {call_fit16*1e3:.3f}"
    )
    return "\n    " + "\n    ".join(lines)

# --- Check 8: row tiling ------------------------------------------------------------


def check_tiling(config):
    npol = 8
    rows = npol * K.EPISODES_PER_POLICY
    exe = K.build_policy(config["algo"], config["config"], config["pop"], rows)
    rng = np.random.default_rng(8)
    obs = f32(rng.uniform(-1.0, 1.0, size=(npol, K.NUM_OBS)))
    tiled = np.concatenate(
        [obs[p : p + 1].repeat(K.EPISODES_PER_POLICY, axis=0) for p in range(npol)], axis=0
    )
    got = K.run_policy(exe, config["pop"], tiled)
    for p in range(npol):
        block = got[p * 3 : p * 3 + 3]
        assert np.array_equal(block[0], block[1]) and np.array_equal(block[0], block[2]), (
            f"policy {p} rows differ: {block}"
        )

    perturbed = tiled.copy()
    target = 3 * 3 + 1
    perturbed[target] = perturbed[target] + F32(0.5)
    got2 = K.run_policy(exe, config["pop"], perturbed)
    untouched = [r for r in range(rows) if r != target]
    max_untouched = float(np.abs(got2[untouched] - got[untouched]).max())
    assert max_untouched == 0.0, max_untouched
    assert not np.array_equal(got2[target], got[target]), "perturbed row did not change"
    return (
        f"identical within triples; untouched-row max delta {max_untouched:.0f}; "
        f"perturbed-row max delta {float(np.abs(got2[target]-got[target]).max()):.6f}"
    )


# --- main ---------------------------------------------------------------------------


def main():
    print("building pipelines ...", flush=True)
    config = V.make_pipeline(8, seed=7)
    config16 = V.make_pipeline(16, seed=7)
    print(f"pipeline pop shapes: {[p.shape for p in config['pop']]}", flush=True)

    run_check("1 obs parity", check_obs)
    run_check("2 reward parity", check_reward)
    run_check("3 termination parity", check_termination)
    run_check("4 first-done/return/mask", check_semantics)
    run_check("5 fitness", check_fitness)
    run_check("6 two generations", lambda: check_generations(config))
    run_check("7 timings", lambda: check_timings(config, config16))
    run_check("8 row tiling", lambda: check_tiling(config))

    print("\n===== SUMMARY =====", flush=True)
    n_pass = 0
    for name, status, detail in RESULTS:
        print(f"{status:4s} {name}: {detail}", flush=True)
        n_pass += status == "PASS"
    print(f"{n_pass}/{len(RESULTS)} passed", flush=True)


if __name__ == "__main__":
    main()
