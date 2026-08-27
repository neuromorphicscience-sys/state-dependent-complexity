mods=['numpy','pandas','scipy','sklearn','matplotlib','torch','pynwb']
import importlib,json
out={}
for m in mods:
    try:
        z=importlib.import_module(m);out[m]={'ok':True,'version':getattr(z,'__version__','unknown')}
    except Exception as e:out[m]={'ok':False,'error':repr(e)}
print(json.dumps(out,indent=2))
if not all(v['ok'] for v in out.values()): raise SystemExit(2)
