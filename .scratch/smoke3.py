import sys, time
sys.path.insert(0, "examples/locomotion")
import numpy as np, evoxc
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
pop = [k._host(a) for a in state.pop]
species_leaves = [k._host(a) for a in evoxc.tree_leaves(state.species)]
rows = NP * k.EPISODES_PER_POLICY
cache = k.ExecutableCache()

t0 = time.perf_counter()
pol = cache.get(("policy", rows, "numpy", "cpu:0"), lambda: k.build_policy(algo, config, pop, rows))
print("policy compiled %.4f s" % (time.perf_counter() - t0))
rng = np.random.default_rng(11)
obs = (rng.normal(size=(rows, 45)) * 0.3).astype(np.float32)
actions = k.run_policy(pol, pop, obs)
print("actions", actions.shape, actions.dtype)
obs2 = obs.copy(); obs2[1] = obs2[0]; obs2[2] = obs2[0]
a2 = k.run_policy(pol, pop, obs2)
print("tile identical:", np.array_equal(a2[0], a2[1]), np.array_equal(a2[0], a2[2]))
obs3 = obs2.copy(); obs3[2] += 1.5
a3 = k.run_policy(pol, pop, obs3)
print("row isolation:", np.array_equal(a3[0], a2[0]), np.array_equal(a3[1], a2[1]), not np.array_equal(a3[2], a2[2]))

t0 = time.perf_counter()
tell = cache.get(("tell", NP, "numpy", "cpu:0"), lambda: k.build_tell(algo, config, pop, species_leaves, NP))
print("tell compiled %.4f s" % (time.perf_counter() - t0))
fit = rng.normal(size=(NP,)).astype(np.float32)
res = k.run_tell(tell, randkey=state.randkey, pop_arrays=pop, generation=state.generation, fitness=fit, species_leaves=species_leaves)
print("randkey", res["randkey"], res["randkey"].dtype, "gen", res["generation"], res["generation"].dtype)
print("pop shapes", [a.shape for a in res["pop"]])
print("species", type(res["species"]).__name__, [np.shape(a) for a in evoxc.tree_leaves(res["species"])])
print("changed", [not np.array_equal(a, b) for a, b in zip(res["pop"], pop)])
print("hits/misses", cache.hits, cache.misses)
