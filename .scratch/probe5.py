import inspect
import tensorneat.pipeline as p
src = inspect.getsource(p)
for i, line in enumerate(src.splitlines(), 1):
    if "species_tree" in line or "def setup" in line or "def step" in line or "def auto_run" in line:
        print(i, line)
