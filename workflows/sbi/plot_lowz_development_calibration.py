"""Render the November CFM score receipt and its six precomputed visual checks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def render(source, dest):
    data = json.loads(source.read_text())
    phases = data["independent_units"]
    rows = [r for r in data["rows"] if r["weighting"] == "cells"]
    assert data["draws"] == 32 and len(set(data["paths"].values())) == 192
    fields = ["Density", "λ₁", "λ₂", "λ₃"]
    colors = ["#2167a2", "#c35812", "#20815c"]
    dest.mkdir(parents=True, exist_ok=True)
    with PdfPages(dest/"calibration.pdf") as pdf:
        def save(fig, name):
            fig.savefig(dest/(name+".png"), dpi=145)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True, layout="constrained")
        for axrow, low in zip(axes, [False, True]):
            for j, ax in enumerate(axrow):
                for p, col in zip(phases, colors):
                    rr = [r for r in rows if r["phase"] == p and (not low or r["z"][0] == .01)]
                    h = np.sum([r["rank_counts"][j] for r in rr], axis=0)
                    assert h.size == 33 and h.sum() == sum(r["cells"] for r in rr)
                    ax.stairs(100*h/h.sum(), np.arange(34)-.5, label=p, color=col)
                ax.axhline(100/33, color=".35", ls=":")
                ax.set_title(fields[j])
                ax.set_xticks([0,8,16,24,32]); ax.set_xlabel("Truth rank")
            axrow[0].set_ylabel(("Low z" if low else "Full range")+" · cells (%)")
        axes[0,-1].legend(fontsize=8)
        axes[0,0].set_ylim(bottom=0)
        fig.suptitle("Single-cell SBC-style ranks · 32 draws · dotted line = uniform reference\nBoth caps; full z = [0.01, 0.55), low z = [0.01, 0.15)")
        save(fig, "ranks")

        fig, aa = plt.subplots(1,4, figsize=(13,3.7), sharey=True, layout="constrained")
        for j,ax in enumerate(aa):
            for name,rr,style in [("All cells",rows,"-"),("Low z",[r for r in rows if r["z"][0]==.01],"--")]:
                h=np.sum([r["rank_counts"][j] for r in rr],axis=0)
                u=np.arange(17)
                ax.plot(100*(33-2*u)/33,[100*h[k:33-k].sum()/h.sum() for k in u],style,label=name)
            ax.plot([0,100],[0,100],":",color=".35"); ax.axvline(90,color=".75",lw=.8)
            ax.set(title=fields[j],xlabel="Nominal rank coverage (%)")
        aa[0].set_ylabel("Empirical coverage (%)"); aa[-1].legend()
        fig.suptitle("Single-cell central rank coverage · linearly interpolated between attainable rank levels")
        save(fig,"coverage")

        fig,aa=plt.subplots(2,3,figsize=(11,6.7),sharex=True,sharey=True,layout="constrained")
        for j,(reg,label) in enumerate([("two_cores","2 core means"),("eight_subregions","8 subregion means")]):
            for ax,p in zip(aa[j],phases):
                rr=[r for r in data["joint"] if r["phase"]==p and r["region"]==reg]
                for width,style in [("0.25","-"),("1.0","--")]:
                    ax.plot(rr[0]["alpha"],np.mean([r["curves"][width] for r in rr],axis=0),style,label="Ref. SD "+width)
                ax.plot([0,1],[0,1],":",color=".35");ax.set_title(p+" · "+label)
                ax.set_xlabel("Nominal credibility")
            aa[j,0].set_ylabel("Empirical coverage")
        aa[0,-1].legend(fontsize=8); fig.suptitle("Density TARP · regional summaries, not single-cell calibration")
        save(fig,"tarp")

        fig,aa=plt.subplots(1,4,figsize=(13,3.8),sharey=True,layout="constrained")
        for j,(ax,label) in enumerate(zip(aa,["Void","Sheet","Filament","Knot"])):
            for p,col in zip(phases,colors):
                rr=[r["classes"][j]["reliability"] for r in rows if r["phase"]==p]
                n,x,y=[np.sum([r[k] for r in rr],axis=0) for k in ("weight","sum_probability","sum_truth")]
                ok=n>0
                ax.plot(x[ok]/n[ok],y[ok]/n[ok],color=col,lw=1,label=p)
                ax.scatter(x[ok]/n[ok],y[ok]/n[ok],s=4+7*np.log10(n[ok]),color=col)
            ax.plot([0,1],[0,1],":",color=".35");ax.set(title=label,xlabel="Mean predicted probability",xlim=(-.02,1.02))
        aa[0].set_ylabel("Observed frequency"); aa[-1].legend(fontsize=8)
        fig.suptitle("Single-cell T-web reliability · threshold 0.2 · larger markers = more scored cells")
        save(fig,"reliability")

        for p in phases:
            for cap in ["NGC","SGC"]:
                im=plt.imread(source.parent/f"{p}_{cap}.png")
                fig,ax=plt.subplots(figsize=(9,13),layout="constrained")
                ax.imshow(im);ax.axis("off");ax.set_title(f"{p} · {cap} · registered core slices")
                pdf.savefig(fig);plt.close(fig)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("source",type=Path);cli.add_argument("dest",type=Path)
    args=cli.parse_args();render(args.source,args.dest)
