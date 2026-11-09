import numpy as np, evoxc

def kernel(node_keys, obs):
    print("type", type(node_keys))
    print("shape attr", getattr(node_keys, "shape", "MISSING"))
    print("spec", getattr(node_keys, "spec", "MISSING"))
    print("dtype", getattr(node_keys, "dtype", "MISSING"))
    print("dir", [n for n in dir(node_keys) if not n.startswith("_")])
    return obs

exe = evoxc.build(
    kernel,
    evoxc.TensorSpec((8, 80), evoxc.int32, name="node_keys"),
    evoxc.TensorSpec((8, 45), evoxc.float32, name="obs"),
    backend="numpy",
    device=evoxc.Device("cpu", 0),
)
print("built")
