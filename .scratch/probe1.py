import numpy as np
import evoxc
from tensorneat import Pipeline, algorithm, genome, problem


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
print("state type:", type(state))
print("pop:", type(state.pop), len(state.pop) if hasattr(state.pop, "__len__") else None)
for i, a in enumerate(evoxc.tree_leaves(state.pop)):
    print("  pop leaf", i, getattr(a, "shape", None), getattr(a, "dtype", None))
print("species type:", type(state.species))
sp_leaves = evoxc.tree_leaves(state.species)
sp_tree = evoxc.tree_structure(state.species)
print("species tree:", sp_tree)
print("n species leaves:", len(sp_leaves))
for i, a in enumerate(sp_leaves):
    print("  species leaf", i, getattr(a, "shape", None), getattr(a, "dtype", None))
print("randkey:", state.randkey, getattr(state.randkey, "shape", None), getattr(state.randkey, "dtype", None))
print("generation:", state.generation, type(state.generation))
config = pipe._static_config(state)
print("config type:", type(config))
print("config attrs:", [n for n in dir(config) if not n.startswith("__")])
print("state attrs:", [n for n in dir(state) if not n.startswith("__")])
print("register:", config.register.__doc__[:1500] if hasattr(config.register, "__doc__") else config.register)
