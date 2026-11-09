import evoxc
names = ["bool_", "int32", "float32", "int64", "dtype", "SymbolicTensor", "Executable", "BoundExecutable", "TensorSpec", "Device"]
for n in names:
    print(n, hasattr(evoxc, n), getattr(evoxc, n, None))
