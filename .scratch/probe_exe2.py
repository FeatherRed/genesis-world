import evoxc, inspect
cls = evoxc.pipeline.Executable
print([n for n, v in vars(cls).items() if isinstance(v, property)])
print("mro", cls.__mro__)
print("slots?", getattr(cls, "__slots__", None))
