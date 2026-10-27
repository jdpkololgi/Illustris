#!/usr/bin/env python3
"""Title-free A--E talk bodies and editable build layers from verified frozen inputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,PathPatch,Patch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap,TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
from matplotlib.text import Text
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010');OUT=Path(os.environ.get('TOKYO_REVISION_OUTPUT',str(ROOT/'revisions_20261018')));DATA=OUT/'data';REPO=Path('/global/homes/d/dkololgi/TNG/Illustris')
W,H=1920,900;WHITE='#F5F5F5';GREY='#C8CDD4';EDGE='#657080';DARK='#0B0E14';C=['#9CF6D5','#5B8FF9','#E85D9E','#F2C14E'];N=['Void','Wall','Filament','Cluster']
FONTDIR=REPO/'assets/fonts'
for p in FONTDIR.glob('IBMPlexSans-*.ttf'):font_manager.fontManager.addfont(p)
FONT='IBM Plex Sans';assert font_manager.findfont(FONT,fallback_to_default=False)
plt.rcParams.update({'font.family':FONT,'font.size':24*.72,'text.color':WHITE,'axes.labelcolor':WHITE,'xtick.color':GREY,'ytick.color':GREY,'axes.edgecolor':EDGE,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'black','axes.unicode_minus':True})
AUDIT=json.loads((OUT/'SOURCE_AUDIT.json').read_text());QA=[];CATALOG=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fig():return plt.figure(figsize=(19.2,9),dpi=100,facecolor='black')
def text(f,x,y,s,px=34,color=WHITE,ha='center',**kw):return f.text(x/W,1-y/H,s,fontsize=px*.72,color=color,ha=ha,va='center',linespacing=1.25,**kw)
def box(f,x,y,w,h,color=EDGE):
 p=FancyBboxPatch((x/W,1-(y+h)/H),w/W,h/H,boxstyle='round,pad=0.004,rounding_size=0.01',transform=f.transFigure,facecolor=DARK,edgecolor=color,lw=1.5,zorder=0);f.add_artist(p);return p
def arrow(f,a,b,style='-|>',dashed=False):
 p=FancyArrowPatch((a[0]/W,1-a[1]/H),(b[0]/W,1-b[1]/H),transform=f.transFigure,arrowstyle=style,mutation_scale=18,lw=1.4,color=GREY,linestyle='--' if dashed else '-');f.add_artist(p)
def polyline(f,xy,color=GREY,lw=1.4,ls='-'):
 f.add_artist(Line2D(np.asarray(xy)[:,0]/W,1-np.asarray(xy)[:,1]/H,transform=f.transFigure,color=color,lw=lw,ls=ls))
def axat(f,x,y,w,h):
 a=f.add_axes([x/W,1-(y+h)/H,w/W,h/H],facecolor='black');a.tick_params(labelsize=28*.72,pad=7,length=4)
 for side in ('top','right'):a.spines[side].set_visible(False)
 return a
def export(f,group,name,title,preview=True,transparent=False):
 dest=OUT/group;dest.mkdir(exist_ok=True)
 f.canvas.draw();r=f.canvas.get_renderer();boxes=[]
 for t in f.findobj(Text):
  if t.get_visible() and t.get_text().strip():
   b=t.get_window_extent(r)
   assert b.x0>=-1 and b.y0>=-1 and b.x1<=W+1 and b.y1<=H+1,(name,t.get_text(),b.bounds)
   boxes.append({'text':t.get_text(),'bbox':list(b.bounds),'font_px':t.get_fontsize()/72*100})
 for ext in ('png','pdf','svg'):f.savefig(dest/(name+'.'+ext),dpi=100,facecolor='none' if transparent else 'black',transparent=transparent)
 assert Image.open(dest/(name+'.png')).size==(W,H)
 assert b'/FontFile2' in (dest/(name+'.pdf')).read_bytes()
 assert '<text' in (dest/(name+'.svg')).read_text()
 if preview:
  body=Image.open(dest/(name+'.png')).convert('RGB');full=Image.new('RGB',(1920,1080),'black');full.paste(body,(0,180));dr=ImageDraw.Draw(full);ft=ImageFont.truetype(str(FONTDIR/'IBMPlexSans-Bold.ttf'),48);dr.text((960,85),title,font=ft,fill=WHITE,anchor='mm');full.save(dest/(name+'_preview_1920x1080.png'));full.resize((960,540),Image.Resampling.LANCZOS).save(dest/(name+'_preview_half.png'));body.resize((960,450),Image.Resampling.LANCZOS).save(dest/(name+'_body_half.png'))
  (dest/(name+'_TITLE.txt')).write_text(title+'\n')
 QA.append({'name':group+'/'+name,'text_bounds_pass':True,'min_font_px':min(q['font_px'] for q in boxes),'text':boxes})
 CATALOG.append({'group':group,'name':name,'title':title,'transparent':transparent,'preview':preview});plt.close(f)
# A: diagram, no fabricated numerical scatter and no posterior arrow.
def paired(level=3,only=None):
 f=fig()
 if (only is None and level>=1) or only=='branches':
  box(f,650,70,620,100);text(f,960,120,'Abacus matter + halos',40)
  arrow(f,(820,175),(480,250));arrow(f,(1100,175),(1440,250))
  box(f,145,260,670,110,C[1]);text(f,480,315,'Tidal calculation',38)
  box(f,1105,260,670,110,C[0]);text(f,1440,315,'Galaxy + observing model',38)
  arrow(f,(480,375),(480,440));arrow(f,(1440,375),(1440,440))
 if (only is None and level>=2) or only=='outputs':
  box(f,145,450,670,180,C[1]);text(f,480,492,'Known local eigenvalues',36);text(f,480,544,r'$\boldsymbol{\lambda}_g = (\lambda_1, \lambda_2, \lambda_3)$',38,color=C[1]);text(f,480,593,'Smoothed matter field; host-linked target',26,color=GREY)
  box(f,1105,450,670,180,C[0]);text(f,1440,490,'Mock observed context',36);text(f,1440,546,'Galaxy population, selection',28,color=GREY);text(f,1440,590,'and redshift-space positions',28,color=GREY)
  text(f,480,665,'R = 7 h$^{-1}$ Mpc',28,color=GREY)
 if (only is None and level>=3) or only=='pairing':
  polyline(f,[(480,690),(480,715),(1440,715),(1440,690)])
  text(f,960,755,'Paired examples: context + known local target',36)
  text(f,960,801,'From the same simulated realisation · used to train and test inference',26,color=GREY)
 return f
AT='Simulations give us both sides of the problem'
export(paired(),'A_paired_simulations','A_body',AT)
for i in (1,2,3):export(paired(i),'A_paired_simulations',f'A_build_{i}',AT,preview=False)
for s in ('branches','outputs','pairing'):export(paired(only=s),'A_paired_simulations','A_component_'+s,AT,preview=False,transparent=True)
# B: exact authored contour segments, no new projection or point recolouring.
paper=json.loads((DATA/'paperI_contours.json').read_text())
def paper_diagram(f):
 for y,h,label in [(110,115,'Galaxy-derived graph inputs\n10 Delaunay graph metrics'),(300,90,'Classifier'),(455,90,'Class probabilities')]:
  box(f,185,y,435,h,C[1] if y==300 else EDGE);text(f,402,y+h/2,label,30 if y==110 else 36)
 arrow(f,(402,230),(402,292));arrow(f,(402,395),(402,448))
 box(f,185,600,435,70,C[0]);text(f,402,635,'Simulation tidal field → training labels',24)
 polyline(f,[(180,635),(110,635),(110,345)],ls='--');arrow(f,(110,345),(178,345),dashed=True)
 text(f,90,483,'Supervision only',24,color=GREY,rotation=90)
def paper_panel(f):
 text(f,1320,80,'Saved predicted-class contours',34)
 a=axat(f,1040,125,540,440)
 for p in paper['paths']:
  q=np.array(p['segments_data']);verts=[];codes=[];previous=None
  for u,v in q:
   if previous is None or not np.allclose(u,previous,atol=1e-10,rtol=0):verts.append(u);codes.append(MPath.MOVETO)
   verts.append(v);codes.append(MPath.LINETO);previous=v
  path=MPath(verts,codes);a.add_patch(PathPatch(path,facecolor=C[p['class_index']],edgecolor='none',alpha=.25));a.add_patch(PathPatch(path,facecolor='none',edgecolor=C[p['class_index']],lw=1.5))
 a.set(xlim=(-.317,7.302),ylim=(1.48,16.255),xticks=[0,2,4,6],yticks=[2,6,10,14]);a.set_xlabel('UMAP 1',fontsize=32*.72,labelpad=9);a.set_ylabel('UMAP 2',fontsize=32*.72,labelpad=11)
 for k,name in enumerate(N):
  f.add_artist(Line2D([1650/W],[1-(215+83*k)/H],transform=f.transFigure,marker='s',color=C[k],ls='',ms=12));text(f,1685,215+83*k,name,30,color=C[k],ha='left')
 text(f,1320,680,'Same saved projection · arbitrary embedding coordinates',24,color=GREY)
BT='From web labels to continuous environments'
f=fig();paper_diagram(f);paper_panel(f);text(f,960,735,'Class probabilities describe uncertainty in a web label',34);text(f,960,780,'Next: a joint distribution of continuous tidal eigenvalues',34);text(f,960,825,'Kololgi et al. (2026), RASTI · doi:10.1093/rasti/rzag025',24,color=GREY);export(f,'B_paperI','B_body',BT)
for key,fn in [('diagram',paper_diagram),('class_panel',paper_panel)]:
 f=fig();fn(f);export(f,'B_paperI','B_component_'+key,BT,preview=False,transparent=True)
# C: same context, inferred actual joint draws; no generative arrow.
c=dict(np.load(DATA/'local_posterior.npz'));counts=np.bincount((c['draws']>.2).sum(1),minlength=4);probs=counts/512;chosen=c['chosen_draw_indices'];xy=c['observed_xyz_relative_mpc'][:,:2]
def context(f):
 text(f,455,72,'Observed mock context',36);text(f,455,112,'Redshift space · Planck18 comoving Mpc',24,color=GREY)
 a=axat(f,180,155,550,500);a.scatter(xy[:,0],xy[:,1],s=12,c=GREY,alpha=.62,lw=0);a.scatter([0],[0],s=170,facecolors='none',edgecolors=C[3],lw=1.5);a.scatter([0],[0],s=25,c=C[3],zorder=5)
 a.set(xlim=(-40,40),ylim=(-40,40),aspect='equal',xticks=[-40,0,40],yticks=[-40,0,40]);a.set_xlabel('Δx [Mpc]',fontsize=34*.72,labelpad=9);a.set_ylabel('Δy [Mpc]',fontsize=34*.72,labelpad=9)
def infer(f):
 arrow(f,(820,400),(1040,400));text(f,930,345,'Infer under\nmodel M',28,color=GREY)
def cards(f,bar=False):
 text(f,1475,75,'Plausible local tidal environments',36)
 for k,y in [(0,155),(1,420)]:
  box(f,1100,y,730,225,C[k]);text(f,1130,y+42,N[k],42,color=C[k],ha='left')
  if not bar:text(f,1820,y+42,f'P({N[k].lower()}) = {probs[k]:.1%}',34,color=C[k],ha='right')
  text(f,1130,y+101,'Example posterior draw',28,color=GREY,ha='left')
  trip=c['draws'][chosen[k]];label=r'$\boldsymbol{\lambda}_g$ = ('+', '.join(f'{v:.3f}'.replace('-','−') for v in trip)+')';text(f,1475,y+157,label,38,color=C[k])
  text(f,1130,y+203,'Ordered, dimensionless eigenvalues',24,color=GREY,ha='left')
 if bar:
  a=f.add_axes([1100/W,1-715/H,750/W,30/H],facecolor='black');start=0
  for k,p in enumerate(probs):
   if p:a.barh([0],[p],left=[start],height=.8,color=C[k]);start+=p
  a.set_xlim(0,1);a.axis('off');text(f,1100,669,f'Void {probs[0]:.1%}',26,color=C[0],ha='left');text(f,1850,669,f'Wall {probs[1]:.1%}',26,color=C[1],ha='right')
 else:text(f,1475,688,'λ₁ ≤ λ₂ ≤ λ₃ · class threshold λth = 0.2',28,color=GREY)
def cfooter(f):
 text(f,960,765,'One observed context can support more than one tidal class.',36)
 text(f,960,815,'Mock posterior example · triplets are example draws; percentages summarise classes.',26,color=GREY)
CT='One galaxy context, several plausible environments'
for bar in (False,True):
 f=fig();context(f);infer(f);cards(f,bar);cfooter(f);export(f,'C_ambiguity','C_probability_bar_body' if bar else 'C_body',CT)
f=fig();context(f);export(f,'C_ambiguity','C_build_1_context',CT,preview=False)
f=fig();context(f);infer(f);cards(f);cfooter(f);export(f,'C_ambiguity','C_build_2_inference',CT,preview=False)
for key,fn in [('context',context),('inference_arrow',infer),('posterior_cards',cards),('footer',cfooter)]:
 f=fig();fn(f);export(f,'C_ambiguity','C_component_'+key,CT,preview=False,transparent=True)
# D: saved alpha/ECP and saved max-deviation envelope, all5001points.
t=dict(np.load(DATA/'tarp_arrays.npz'));f=fig();text(f,960,80,'Joint-posterior check in mocks · TARP',34)
a=axat(f,230,155,1100,480);a.fill_between(t['alpha'],t['envelope_lower'],t['envelope_upper'],color=C[1],alpha=.42,lw=0);a.plot(t['alpha'],t['expected_coverage_probability'],color=C[0],lw=2.5,zorder=4);a.plot([0,1],[0,1],ls=(0,(6,4)),color=WHITE,lw=1.1,zorder=5)
a.set(xlim=(0,1),ylim=(0,1),xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6));a.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));a.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));a.set_xlabel('Probability assigned to test region',fontsize=34*.72,labelpad=15);a.set_ylabel('Fraction of mock truths inside',fontsize=34*.72,labelpad=15)
f.legend(handles=[Line2D([],[],color=C[0],lw=2.5,label='Mock test'),Line2D([],[],color=WHITE,ls='--',label='Perfect agreement'),Patch(facecolor=C[1],alpha=.42,label='Bootstrap reference\nenvelope')],loc='center left',bbox_to_anchor=(.735,.57),frameon=False,fontsize=28*.72,handlelength=1.6,labelspacing=1.5)
text(f,960,765,'50,000 mock galaxies · held out from training',34);text(f,960,815,'Lemos et al. (2023) · arXiv:2302.03026',24,color=GREY);export(f,'D_tarp','D_body','Do the probabilities mean what they say?')
# E: original data/palette/cuts, clarify the actual coordinate placement.
d=dict(np.load(DATA/'tidal_overlay.npz'));dm=dict(np.load(DATA/'density_tracers.npz'));bounds=dm['extent'];lim=1.75
f=fig();positions=[(165,105),(1035,105),(165,405),(1035,405)]
for k,(x,y) in enumerate(positions):
 a=axat(f,x,y,640,240)
 if k<3:im=a.imshow(d['eigenvalues'][...,k].T,origin='lower',extent=bounds,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),interpolation='nearest')
 else:a.imshow(d['classes'].T,origin='lower',extent=bounds,cmap=ListedColormap(C),vmin=-.5,vmax=3.5,alpha=.78,interpolation='nearest')
 a.scatter(d['xy'][:,0],d['xy'][:,1],s=5.2,c='white',edgecolors='black',lw=.3,zorder=3);a.set(xlim=bounds[:2],ylim=bounds[2:],aspect='equal',xticks=[-400,-300],yticks=[100,150]);a.set_title(['λ₁','λ₂','λ₃','T-web · λth = 0.2'][k],loc='left',fontsize=36*.72,pad=10)
 if k<2:a.tick_params(labelbottom=False)
 else:a.set_xlabel('x [h$^{-1}$ Mpc]',fontsize=32*.72,labelpad=8)
 if k%2==0:a.set_ylabel('y [h$^{-1}$ Mpc]',fontsize=32*.72,labelpad=8)
 else:a.tick_params(labelleft=False)
cax=f.add_axes([1780/W,1-625/H,22/W,410/H]);cb=f.colorbar(im,cax=cax,ticks=[-1.5,0,1.5]);cb.ax.tick_params(labelsize=28*.72,pad=8);cax.set_title('λ',fontsize=32*.72,pad=12);text(f,1740,420,'Dimensionless',26,color=GREY,rotation=90)
f.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(C,N)],loc='center',bbox_to_anchor=(.71,1-770/H),ncol=4,frameon=False,fontsize=27*.72,handlelength=.7,columnspacing=.8)
text(f,960,820,'Real-space field and true mock positions · observed-success selection · ph002 · R = 7 h$^{-1}$ Mpc',26,color=GREY)
export(f,'E_coordinates','E_tidal_four_panel_body','The simulated tidal field and its selected tracers')
f=fig();text(f,390,100,'Simulated matter',36);text(f,390,145,'Real-space density',28,color=GREY);text(f,960,100,'Galaxy + observing model',36);text(f,1550,100,'Selected mock galaxies',36);text(f,1550,145,'Shown at true mock positions',28,color=GREY)
a=axat(f,140,225,500,250);im=a.imshow(dm['rho_over_mean'].T,origin='lower',extent=bounds,cmap='viridis',vmin=0,vmax=4,interpolation='bilinear');b=axat(f,1300,225,500,250);b.scatter(dm['xy'][:,0],dm['xy'][:,1],s=6,c=WHITE,lw=0)
for z in (a,b):z.set(xlim=bounds[:2],ylim=bounds[2:],aspect='equal',xticks=[-400,-300],yticks=[100,150]);z.set_xlabel('x [h$^{-1}$ Mpc]',fontsize=32*.72,labelpad=10)
a.set_ylabel('y [h$^{-1}$ Mpc]',fontsize=32*.72,labelpad=8);b.tick_params(labelleft=False)
box(f,770,195,380,310)
for yy,ss in zip([250,320,390,460],['Galaxy population','Brightness limit','Sky coverage','Redshift success']):text(f,960,yy,ss,34,color=GREY)
arrow(f,(665,350),(750,350));arrow(f,(1170,350),(1270,350))
cax=f.add_axes([165/W,1-615/H,450/W,20/H]);cb=f.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,2,4]);cb.ax.tick_params(labelsize=28*.72,pad=6);cb.set_label('ρ / ρ̄ (dimensionless)',fontsize=30*.72,labelpad=8)
text(f,1550,610,'718 selected mock galaxies',32)
text(f,960,750,'Observed-success selection is applied; the dots retain true mock positions.',32)
text(f,960,810,'ph002 · DESI/c000 distances · R = 7 h$^{-1}$ Mpc · slab = 6.766 h$^{-1}$ Mpc',26,color=GREY)
export(f,'E_coordinates','E_survey_selection_body','Train on what the survey can observe')
# Separate context component doubles as corrected coordinate-caption asset.
shutil.copy2(OUT/'C_ambiguity/C_component_context.png',OUT/'E_coordinates/E_ambiguity_context.png')
shutil.copy2(OUT/'C_ambiguity/C_component_context.pdf',OUT/'E_coordinates/E_ambiguity_context.pdf')
shutil.copy2(OUT/'C_ambiguity/C_component_context.svg',OUT/'E_coordinates/E_ambiguity_context.svg')
(OUT/'qa/text_bounds.json').write_text(json.dumps(QA,indent=2)+'\n');(OUT/'FIGURES.json').write_text(json.dumps(CATALOG,indent=2)+'\n')
(OUT/'scripts').mkdir(exist_ok=True)
for name in ['tokyo_revision_sources_20261018.py','tokyo_revisions_20261018.py']:shutil.copy2(REPO/'workflows/presentation'/name,OUT/'scripts'/name)
print('Rendered',len(CATALOG),'bodies/layers; IBM Plex Sans; unchanged scientific arrays')
