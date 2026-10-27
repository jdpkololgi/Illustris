import json,hashlib
from pathlib import Path
c=Path(__file__).resolve().parent
m=json.loads((c/"SOURCE_FREEZE.json").read_text())
for name,h in m["source_sha256"].items():
 if hashlib.sha256((Path(m["source_root"])/name).read_bytes()).hexdigest()!=h:raise RuntimeError("Source changed: "+name)
print("SOURCE_FREEZE_VERIFIED",flush=True)
