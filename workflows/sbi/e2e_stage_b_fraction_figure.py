"""Bounded rendering only, from the independently validated Stage B receipt."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/calibration_investigation_20261006_v1/b1b2')
report=json.loads((ROOT/'REVIEW.json').read_text())
assert report['passed'] and report['checks']==1184
out=Path(__file__).resolve().parents[2]/'docs/figures/e2e_stage_b_20261006/fraction_coverage.png'
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
styles=(('original','#2563a6','o'),('corrected','#d07a26','s'))
for i,weight in enumerate(('volume','galaxy')):
    for j,row in enumerate(report['headline']):
        ax=axes[i,j]
        for offset,(method,color,marker) in zip((-.1,.1),styles):
            q=row['methods'][method]['regional'][weight]
            ax.scatter(np.arange(4)+offset,q['coverage']['0.9'],color=color,marker=marker,label=method.capitalize())
        ax.axhline(.9,ls=':',color='#333333')
        ax.set(title=f"{row['phase']} · {weight}-weighted fractions",xticks=np.arange(4),
               xticklabels=['Void','Wall','Filament','Knot'],ylim=(0,1.04))
        ax.grid(axis='y',alpha=.15)
        if j==0:ax.set_ylabel('Rank coverage')
axes[0,0].legend(fontsize=8,loc='lower left')
fig.suptitle('Regional class-fraction coverage at threshold 0.2')
fig.text(.5,.015,'Randomization-averaged rank inclusion: nominal 90% is the exact tie-aware reference.\n32 cores per phase; equal-core summaries, dependent regions, no population qualification.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.085,1,.95));fig.savefig(out,dpi=160);plt.close(fig)
print(json.dumps(dict(figure=str(out),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    review_sha256=hashlib.sha256((ROOT/'REVIEW.json').read_bytes()).hexdigest())))
