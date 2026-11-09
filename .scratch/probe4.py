import inspect
import tensorneat.pipeline as p
src = inspect.getsource(p)
i = src.find("_generation_kernel")
print(src[i-200:i+6000])
