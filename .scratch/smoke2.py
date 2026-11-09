import sys, time
sys.path.insert(0, "examples/locomotion")
import numpy as np
import evoxc
from tensorneat import Pipeline, algorithm, genome, problem
import go2_neat_kernels as k

NP = 8
class Stub(problem.BaseProblem):
    input_shape = (45,)
    output_shape = (12,)
    def evaluate(self, state, randkey, act_func, params):
        raise NotImplementedError

gen = genome.DefaultGenome(num_inputs=45, num_outputs=12, max_nodes=80, max_conns=640)
algo = algorithm.NEAT(gen, pop_size=NP, species_size=3)
pipe = Pipeline(algo, Stub(), seed=7, backend="numpy", is_detailed=False)
state = pipe.setup()
config = pipe._static_config(state)
print("pop shapes", [np.shape(a) for a in state.pop])
species_leaves = evoxc.tree_leaves(state.species)
print("species leaves", len(species_leaves), [np.shape(a) for a in species_leaves])
print("randkey", state.randkey, np.shape(state.randkey), "generation", state.generation, np.shape(state.generation))

pop = [np.asarray(a) for a in state.pop]
rows = NP * k.EPISODES_PER_POLICY
t0 = time.perf_counter()
pol = k.build_policy(algo, config, pop, rows)
t1 = time.perf_counter()
print("policy compiled", t1 - t0)

rng = np.random.default_rng(11)
obs = rng.normal(size=(rows, 45)).astype(np.float32) * 0.3
actions = k.run_policy(pol, pop, obs)
print("actions", actions.shape, actions.dtype, actions[0][:4])
# row tiling: identical obs on the 3 rows of policy 0 -> identical actions
obs2 = obs.copy()
obs2[1] = obs2[0]; obs2[2] = obs2[0]
a2 = k.run_policy(pol, pop, obs2)
print("tile identical:", np.array_equal(a2[0], a2[1]), np.array_equal(a2[0], a2[2]))
obs3 = obs2.copy(); obs3[2] = obs3[2] + 1.5
a3 = k.run_policy(pol, pop, obs3)
print("perturb one row changes only it:", np.array_equal(a3[0], a2[0]), np.array_equal(a3[1], a2[1]), not np.array_equal(a3[2], a2[2]))

t0 = time.perf_counter()
tell = k.build_tell(algo, config, pop, species_leaves, NP)
t1 = time.perf_counter()
print("tell compiled", t1 - t0)
fitness = rng.normal(size=(NP,)).astype(np.float32)
res = k.run_tell(
    tell,
    randkey=np.asarray(state.randkey),
    pop_arrays=pop,
    generation=np.asarray(state.generation),
    fitness=fitness,
    species_leaves=species_leaves,
)
print("randkey", res["randkey"], res["randkey"].dtype)
print("generation", res["generation"], res["generation"].dtype)
print("pop shapes", [np.shape(a) for a in res["pop"]])
print("species type", type(res["species"]).__name__, len(evoxc.tree_leaves(res["species"])))
print("changed", [not np.array_equal(a, b) for a, b in zip(res["pop"], pop)])
print("species shapes", [np.shape(a) for a in evoxc.tree_leaves(res["species"])])
print("OK")
