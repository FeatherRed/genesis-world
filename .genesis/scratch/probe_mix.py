import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "examples", "locomotion"))
sys.path.insert(0, os.getcwd())
os.environ.setdefault("EVOXC_IREE_COMPILE_ARGS", "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false")

import numpy as np
import evoxc
import genesis as gs
from tests.device_pilot import as_host

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
from go2_neat_env import Go2NeatEnv
from seeding import Perturbation

device = evoxc.Device("rocm", 0)
print("device ok", device, flush=True)

def run(residency, horizon, **kw):
    print(f"--- build {residency} h{horizon}", flush=True)
    env = Go2NeatEnv(pop_size=8, master_seed=0, perturbation=Perturbation(0.05, 0.05, 0.05), horizon=horizon,
                     backend="numpy" if residency == "numpy" else "iree",
                     device=None if residency == "numpy" else device, residency=residency, **kw)
    try:
        print(f"--- evaluate {residency}", flush=True)
        res = env.evaluate(tuple(np.array(l, copy=True) for l in env._trace_pop), 0)
        print(f"--- fitness {as_host(res['fitness'])[:2]} steps {res['steps']} reasons {res['end_reasons']}", flush=True)
    finally:
        print(f"--- close {residency}", flush=True)
        env.close()
        print(f"--- closed {residency}", flush=True)

run("numpy", 25)
run("device", 25)
run("numpy", 40, episode_length_s=0.1)
run("device", 40, episode_length_s=0.1)
print("--- gs.destroy", flush=True)
gs.destroy()
print("--- done", flush=True)
