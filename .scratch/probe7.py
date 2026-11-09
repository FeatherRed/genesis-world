import inspect
from tensorneat import algorithm
print(inspect.getdoc(algorithm.NEAT.prepare_forward))
print("-----")
print(inspect.getdoc(algorithm.NEAT.forward_prepared))
print("-----")
print(inspect.getdoc(algorithm.NEAT.transform))
