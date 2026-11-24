"""Report observed count ratios on identical cap/PHOTSYS pixels (not completeness)."""
import argparse
import json
from pathlib import Path


def transfer(receipt):
    import numpy as np
    q=json.loads(receipt.read_text())
    assert receipt.name=='RESPONSE_PREPARED.json' and q['passed']
    assert q['cap_order']==['SGC','NGC'] and q['shell_edges']==[.01,.15,.25,.35,.45,.55]
    panels=q['common_domain_counts']
    assert set(panels)=={'ph006','ph012','ph013'}
    rows=[]
    for phase,counts in sorted(panels.items()):
        obs,mock=[np.asarray(counts[k]) for k in ('loa','mock')]
        assert obs.shape==mock.shape==(2,5) and (obs>0).all() and (mock>0).all()
        for i,cap in enumerate(q['cap_order']):
            assert counts['common_pixels'][i]>0
            for j,(o,m) in enumerate(zip(obs[i],mock[i])):
                rows.append(dict(phase=phase,cap=cap,z=q['shell_edges'][j:j+2],loa=int(o),mock=int(m),
                    ratio=float(o/m),common_pixels=counts['common_pixels'][i]))
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();rows=transfer(args.receipt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'counts.json').write_text(json.dumps(rows,indent=1,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(8.8,3.8),layout='constrained')
    for ax,cap in zip(axes,['NGC','SGC']):
        for phase in ['ph006','ph012','ph013']:
            r=[v for v in rows if (v['phase'],v['cap'])==(phase,cap)]
            ax.plot([sum(v['z'])/2 for v in r],[v['ratio'] for v in r],marker='o',label=phase)
        ax.axhline(1,ls='--',color='.45',lw=1)
        ax.set(xlabel='Redshift (bin centre)',title=cap,xticks=[.08,.20,.30,.40,.50])
    axes[0].set_ylabel('Loa / mock galaxy count');axes[1].legend()
    fig.suptitle('BGS_BRIGHT counts on common angular support')
    fig.savefig(args.out/'counts.pdf');plt.close(fig)
