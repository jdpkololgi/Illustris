import hashlib,json
from pathlib import Path
import torch
b=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
rows=[]
for p in ['ph000','ph002','ph003','ph004','ph005','ph006']:
 s=json.loads((b/f'p12_oof_summaries/{p}/OOF_SUMMARY_COMPLETE.json').read_text())
 ck=Path(s['checkpoint']); c=torch.load(ck,map_location='cpu',weights_only=False)
 root=Path(s['contract_root'])
 normal=json.loads((root/'transforms/field/field_transform.json').read_text())['normalization']
 scaler=json.loads((root/'transforms/target_scaler.json').read_text())
 sel=root/'transforms/field/selection_manifest.json'
 rows.append(dict(phase=p,checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),checks=dict(normalization=c['normalization']==normal,scaler=c['scaler']==scaler,epoch=c['epoch']==20,membership=c['training_phases']==s['training_phases']),source_contract=c['source_contract'],selection=str(sel),selection_sha256=hashlib.sha256(sel.read_bytes()).hexdigest()))
out=Path('/global/u2/d/dkololgi/TNG/Illustris/docs/evidence/p12/P12A_EMBEDDED_TRANSFORMS_20260924.json')
r=dict(records=rows,pass_checks=all(all(x['checks'].values()) for x in rows),ready_for_desi_canary=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with out.open('x') as f:json.dump(r,f,indent=2)
print(r['pass_checks']);print(rows[-1]['source_contract'])
