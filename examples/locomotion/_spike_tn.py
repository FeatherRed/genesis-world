"""Temporary spike (deleted before commit): drive TensorNEAT NEAT from a host loop.

Two compiled executables, both cached by shape:
  policy: payload (genome) + obs -> actions  (transform + prepare + forward in ONE trace)
  tell  : randkey + pop + generation + fitness + species -> next randkey/generation/pop/species
No Genesis involved: fake observations and fake fitness.
"""

import time

import numpy as np

import evoxc
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common.tools import to_numpy
from tensorneat.genome.utils import GenomeTensors

POP = 8
SAMPLES = 3
OBS_DIM = 45
ACT_DIM = 12


class _Dummy(problem.BaseProblem):
    input_shape = (OBS_DIM,)
    output_shape = (ACT_DIM,)

    def evaluate(self, state, key, act_fn, transformed):
        raise NotImplementedError


def main():
    gen = genome.DefaultGenome(num_inputs=OBS_DIM, num_outputs=ACT_DIM, max_nodes=80, max_conns=640)
    algo = algorithm.NEAT(gen, pop_size=POP, species_size=3)
    pipe = Pipeline(algo, _Dummy(), seed=7, backend="numpy", is_detailed=False)
    state = pipe.setup()
    config = pipe._static_config(state)
    species_tree = evoxc.tree_structure(state.species)
    species_leaves = evoxc.tree_leaves(state.species)
    print("pop leaves:", [tuple(f.shape) for f in state.pop], "species leaves:", len(species_leaves))

    jit = lambda x: np.asarray(to_numpy(x))  # noqa: E731
    pop_np = [jit(f) for f in state.pop]
    rows = POP * SAMPLES

    def policy_kernel(node_keys, node_attrs, conn_keys, conn_attrs, obs):
        payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        transformed = algo.transform(config, payload)
        prepared = algo.prepare_forward(config, transformed, payload, samples=SAMPLES)
        return algo.forward_prepared(config, transformed, prepared, obs)

    specs = [evoxc.TensorSpec(f.shape, f.dtype) for f in pop_np]
    specs.append(evoxc.TensorSpec((rows, OBS_DIM), evoxc.float32))
    t0 = time.perf_counter()
    exe_policy = evoxc.build(policy_kernel, *specs, backend="numpy")
    print(f"policy build {1e3 * (time.perf_counter() - t0):.1f} ms")

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((rows, OBS_DIM)).astype(np.float32)
    t0 = time.perf_counter()
    acts = jit(evoxc.run(exe_policy, *pop_np, obs))
    print(f"policy cold run {1e3 * (time.perf_counter() - t0):.2f} ms -> {acts.shape} {acts.dtype}")

    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        acts_b = jit(evoxc.run(exe_policy, *pop_np, obs))
    dt = (time.perf_counter() - t0) / n
    print(f"policy warm run {1e3 * dt:.3f} ms/call, identical={np.array_equal(acts, acts_b)}")

    # Row tiling: the same obs repeated on a policy's 3 rows must give the same action, and
    # touching one row's obs must change only that row's action.
    rep = np.repeat(obs[:POP], SAMPLES, axis=0)
    rep_acts = jit(evoxc.run(exe_policy, *pop_np, rep))
    print("same obs on a policy's 3 rows -> same action:",
          all(np.array_equal(rep_acts[3 * p], rep_acts[3 * p + 1]) and np.array_equal(rep_acts[3 * p], rep_acts[3 * p + 2])
              for p in range(POP)))
    obs_c = obs.copy()
    obs_c[3 * 2 + 1] += 0.5
    acts_c = jit(evoxc.run(exe_policy, *pop_np, obs_c))
    changed_rows = np.flatnonzero(np.any(acts != acts_c, axis=1)).tolist()
    print("only row 7 changed its action:", changed_rows == [3 * 2 + 1], changed_rows)

    # A payload change must change that policy's actions (perturb an output-node bias).
    pop2 = [f.copy() for f in pop_np]
    pop2[1][0, OBS_DIM:, 2] += 1.0
    acts2 = jit(evoxc.run(exe_policy, *pop2, obs))
    print("perturbed policy 0 changed:", not np.array_equal(acts[0], acts2[0]),
          "| other policies unchanged:", all(np.array_equal(acts[p], acts2[p]) for p in range(1, POP)))

    def tell_kernel(randkey, node_keys, node_attrs, conn_keys, conn_attrs, generation, fitness, *sp):
        payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        species = evoxc.tree_unflatten(list(sp), species_tree)
        current = config.register(pop=payload, generation=generation, species=species, randkey=randkey)
        nxt = algo.tell(current, fitness)
        return [nxt.randkey, nxt.generation, *nxt.pop, *evoxc.tree_leaves(nxt.species)]

    tell_specs = [evoxc.TensorSpec(tuple(state.randkey.shape), state.randkey.dtype)]
    tell_specs += [evoxc.TensorSpec(f.shape, f.dtype) for f in pop_np]
    tell_specs += [evoxc.TensorSpec(tuple(state.generation.shape), state.generation.dtype)]
    tell_specs += [evoxc.TensorSpec((POP,), evoxc.float32)]
    tell_specs += [evoxc.TensorSpec(tuple(l.shape), l.dtype) for l in species_leaves]
    t0 = time.perf_counter()
    exe_tell = evoxc.build(tell_kernel, *tell_specs, backend="numpy")
    print(f"tell build {1e3 * (time.perf_counter() - t0):.1f} ms")

    fit = np.asarray(rng.standard_normal((POP,)), dtype=np.float32)
    outs = evoxc.run(exe_tell, jit(state.randkey), *pop_np, jit(state.generation), fit, *[jit(l) for l in species_leaves])
    print("tell outs:", len(outs), "| gen ->", float(jit(outs[1])), "| species leaves:", len(outs[6:]))
    new_pop = [jit(f) for f in outs[2:6]]
    print("new pop shapes:", [f.shape for f in new_pop],
          "| differs:", [not np.array_equal(a, b) for a, b in zip(pop_np, new_pop)])

    # Generation 2 through the same executables: same shapes -> reuses the compiled artifact.
    acts3 = jit(evoxc.run(exe_policy, *new_pop, obs))
    print("gen2 policy on cached executable:", acts3.shape, "| changed:", not np.array_equal(acts, acts3))
    t0 = time.perf_counter()
    evoxc.run(exe_policy, *new_pop, obs)
    print(f"gen2 policy run {1e3 * (time.perf_counter() - t0):.2f} ms (no rebuild)")


if __name__ == "__main__":
    main()
