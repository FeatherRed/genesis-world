import inspect
import tensorneat.pipeline as p
src = inspect.getsource(p)
lines = src.splitlines()
print("\n".join(lines[955:1060]))
