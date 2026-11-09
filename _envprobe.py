import sys, json, importlib, importlib.metadata as im
out = {}
out['python'] = sys.version
out['exe'] = sys.executable
out['path'] = sys.path
for n in ['genesis','torch','tensorneat','evogp','evoxc','quadrants','tensordict','numpy']:
    try:
        m = importlib.import_module(n)
        out[n] = {'file': getattr(m,'__file__',None), 'ver': getattr(m,'__version__',None)}
    except Exception as e:
        out[n] = {'error': repr(e)}
dver = {}
for d in ['genesis-world','torch','tensorneat','evogp','evoxc','quadrants','tensordict','numpy','rsl-rl-lib']:
    try:
        dver[d] = im.version(d)
    except Exception as e:
        dver[d] = repr(e)
out['dist_versions'] = dver
# evoxc top-level surface
try:
    import evoxc, os
    base = os.path.dirname(evoxc.__file__)
    out['evoxc_files'] = sorted(os.listdir(base))
    out['evoxc_pkgdir'] = base
except Exception as e:
    out['evoxc_probe_error'] = repr(e)
open(r'./_envreport.txt','w').write(json.dumps(out, indent=2, default=str))
