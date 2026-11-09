import numpy as np
import evoxc
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.genome.utils import GenomeTensors
import inspect


class StubProblem(problem.BaseProblem):
    @property
    def input_shape(self):
        return (45,)

    @property
    def output_shape(self):
        return (12,)

    def evaluate(self, state, key, act_fn, transformed):
        raise NotImplementedError


gen = genome.DefaultGenome(num_inputs=45, num_outputs=12, max_nodes=80, max_conns=640)
algo = algorithm.NEAT(gen, pop_size=8, species_size=3)
pipe = Pipeline(algo, StubProblem(), seed=7, backend="numpy", is_detailed=False)
state = pipe.setup()
print(inspect.getsource(pipe._static_config))
config = pipe._static_config(state)
print("config flatten:", config.tree_flatten() if hasattr(config, "tree_flatten") else None)
print("registered keys:", config.registered_keys)
