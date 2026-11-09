import sys
sys.path.insert(0, "examples/locomotion")
import numpy as np, evoxc
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.genome.utils import GenomeTensors

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

def kernel(node_keys, node_attrs, conn_keys, conn_attrs, obs):
    print("kinds", type(node_keys).__name__, type(node_attrs).__name__, type(conn_keys).__name__, type(conn_attrs).__name__)
    print("shapes", node_keys.shape, node_attrs.shape, conn_keys.shape, conn_attrs.shape)
    payload = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
    print("payload", payload.node_keys.shape, payload.node_attrs.shape, payload.conn_keys.shape, payload.conn_attrs.shape)
    print("fields", GenomeTensors._fields)
    transformed = algo.transform(config, payload)
    print("transformed ok")
    return obs

specs = [
    evoxc.TensorSpec((8, 80), evoxc.int32, name="node_keys"),
    evoxc.TensorSpec((8, 80, 4), evoxc.float32, name="node_attrs"),
    evoxc.TensorSpec((8, 640, 2), evoxc.int32, name="conn_keys"),
    evoxc.TensorSpec((8, 640, 1), evoxc.float32, name="conn_attrs"),
    evoxc.TensorSpec((rows, 45), evoxc.float32, name="obs"),
]
exe = evoxc.build(kernel, *specs, backend="numpy", device=evoxc.Device("cpu", 0))
print("built")
