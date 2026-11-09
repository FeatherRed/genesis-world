import evoxc, inspect
src = inspect.getsource(evoxc.run)
open(".scratch/run_src_utf8.txt", "w", encoding="utf-8").write(src)
