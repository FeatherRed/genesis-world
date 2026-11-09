import numpy as np
import evoxc

ROWS = 4


def kernel(a_bool, i, f, x, gravity):
    n = evoxc.logical_not(a_bool)
    e = evoxc.equal(a_bool, 0)
    both = evoxc.logical_and(a_bool, n)
    zero_i = i - i
    i1 = i + 1
    gt = evoxc.greater(i1, 10)
    sel = evoxc.select(gt, zero_i, i1)
    c0 = x[:, 0]
    c2 = x[:, 2]
    stacked = evoxc.stack([c0, c2], axis=-1)
    col = evoxc.reshape(c0, (ROWS, 1))
    conc = evoxc.concatenate([col, col], axis=-1)
    rsum = evoxc.reduce_sum(x * x, axes=-1)
    rmean = evoxc.reduce_mean(evoxc.reshape(rsum, (2, 2)), axes=1)
    ex = evoxc.exp(-(c2 * c2) / 0.25)
    z = evoxc.numpy.zeros((ROWS,), evoxc.int32)
    casted = evoxc.cast(a_bool, evoxc.float32)
    g = evoxc.add(evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(gravity))), 1.0)
    return [n, e, both, sel, casted, stacked, conc, rsum, rmean, ex, z, g]


specs = [
    evoxc.TensorSpec((ROWS,), evoxc.bool_, name="a_bool"),
    evoxc.TensorSpec((ROWS,), evoxc.int32, name="i"),
    evoxc.TensorSpec((ROWS,), evoxc.float32, name="f"),
    evoxc.TensorSpec((ROWS, 3), evoxc.float32, name="x"),
    np.array([0.0, 0.0, -1.0], np.float32),
]
exe = evoxc.build(kernel, *specs, backend="numpy", device=evoxc.Device("cpu", 0))
print("exe type:", type(exe))
print("has __dict__:", hasattr(exe, "__dict__"))
try:
    exe.rows = ROWS
    print("attr attach ok:", exe.rows)
except Exception as e:
    print("attr attach failed:", e)
try:
    print("device:", exe.device, exe.device.kind)
except Exception as e:
    print("no device:", e)

out = evoxc.run(
    exe,
    np.array([True, False, True, False]),
    np.array([9, 10, 11, 0], np.int32),
    np.zeros(ROWS, np.float32),
    np.arange(12, dtype=np.float32).reshape(ROWS, 3),
    np.array([0.0, 0.0, -1.0], np.float32),
)
for i, o in enumerate(out):
    arr = np.asarray(o)
    print(i, arr.shape, arr.dtype, arr.reshape(-1)[:3] if arr.size else arr)

