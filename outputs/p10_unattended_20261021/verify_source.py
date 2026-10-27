from pathlib import Path
import json,hashlib
m=json.loads((Path(__file__).parent/'SOURCE_FREEZE.json').read_text())
for name,h in m['source_sha256'].items():
 if hashlib.sha256((Path(m['source_root'])/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+name)
print('SOURCE_FREEZE_VERIFIED',flush=True)
