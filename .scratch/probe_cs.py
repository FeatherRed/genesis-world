import inspect
from tensorneat.genome import base
src = inspect.getsource(base.BaseGenome._check_schedule_shapes)
open(".scratch/check_shapes.txt", "w", encoding="utf-8").write(src)
import evoxc
print(evoxc.SymbolicTensor.__slots__ if hasattr(evoxc.SymbolicTensor, "__slots__") else [n for n in vars(evoxc.SymbolicTensor)])
print([n for n, v in vars(evoxc.SymbolicTensor).items() if isinstance(v, property)])
