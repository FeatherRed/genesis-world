"""Temporary spike (deleted before commit): list the evoxc op surface."""

import evoxc

names = [n for n in dir(evoxc) if not n.startswith("_")]
print("ALL:", " ".join(sorted(names)))
print()
for n in ("arctan2", "sqrt", "sin", "cos", "tan", "exp", "abs", "clip", "where", "select", "greater", "less",
          "equal", "maximum", "minimum", "mean", "max", "min", "argmax", "sign", "log", "power", "concatenate",
          "stack", "reshape", "broadcast_to", "transpose", "astype", "cast", "reduce_sum", "reduce_max", "zeros",
          "ones", "from_numpy", "constant", "tree_structure", "tree_leaves", "tree_unflatten"):
    print(f"{n:16s} {hasattr(evoxc, n)}")
