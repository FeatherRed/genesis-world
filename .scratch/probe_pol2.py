import sys
sys.path.insert(0, "examples/locomotion")
import numpy as np, evoxc
from tensorneat import Pipeline, algorithm, genome, problem
import go2_neat_kernels as k

NPP = 8
class Stub(problem.BaseProblem):
    input_shape = (45,)
    output_shape = (12,)
    def evaluate(self, state, randkey, act_func, params):
        raise NotImplementedError

gen = genome.DefaultGenome(num_inputs=45, num_outputs=12, max_nodes=80, max_conns=640)
algo = algorithm.NEAT(gen, pop_size=NPP, species_size=3)
pipe = Pipeline(algo, Stub(), seed=7, backend="numpy", is_detailed=False)
state = pipe.setup()
config = pipe._static_config(state)
pop = [np.asarray(a) for a in state.pop]
rows = NPP * 3

print("specs", k._pop_specs(pop))
orig = algo.transform
def patched(state_, payload):
    print("patched transform:", type(payload).__name__, [np.shape(a) if hasattr(a, "shape") else a for a in payload])
    print("  node_keys shape", payload.node_keys.shape, "conn", payload.conn_keys.shape)
    return orig(state_, payload)
algo.transform = patched
try:
    exe = k.build_policy(algo, config, pop, rows)
    print("built")
except Exception as exc:
    print("ERR", type(exc).__name__, exc)
