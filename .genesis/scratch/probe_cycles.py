import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "examples", "locomotion"))
os.environ.setdefault("EVOXC_IREE_COMPILE_ARGS", "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false")

import numpy as np
import genesis as gs

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
from go2_neat_env import Go2NeatEnv
from seeding import Perturbation

for cycle in range(6):
    print(f"--- cycle {cycle} build", flush=True)
    env = Go2NeatEnv(pop_size=8, master_seed=0, perturbation=Perturbation(0.05, 0.05, 0.05), horizon=25, backend="numpy", residency="numpy")
    print(f"--- cycle {cycle} evaluate", flush=True)
    res = env.evaluate(tuple(np.array(l, copy=True) for l in env._trace_pop), 0)
    print(f"--- cycle {cycle} fitness {res['fitness'][:2]} steps {res['steps']}", flush=True)
    print(f"--- cycle {cycle} close", flush=True)
    env.close()
    print(f"--- cycle {cycle} closed", flush=True)

print("--- gs.destroy", flush=True)
gs.destroy()
print("--- done", flush=True)
