"""Actual legacy four-layer GAT: paired smoothing/density transfer experiment.

Topology runs in cosmic_env; metrics in rapids-gnn (CPU fallback explicitly
authorized); training/inference in cosmic_env. All stages use one CPU thread.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse

HOME=Path('/global/u2/d/dkololgi/fmsu_tokyo_20261010')
OLD=HOME/'tng_density_control_20261015'
OUT=HOME/'legacy_gat_density_20261015'
ROOT=Path(__file__).resolve().parents[2]
SEED=20261015

def dump(name,v):
    p=OUT/name;p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def topology(arm):
    import gudhi
    OUT.mkdir(exist_ok=True)
    if arm in ['dense','thin']:
        cat=np.load(OLD/'catalogue.npz');f=np.load(OLD/'features.npz')
        ids=f['dense_tracer_ids' if arm=='dense' else 'thin_20261015_tracer_ids']
        pos=cat['pos_hmpc'][ids]/.6774
        alpha=1.5*((205/.6774)**3/len(pos))**(1/3)
        query=np.arange(len(pos));targetid=cat['subhalo_id'][ids]
    else:
        import fitsio
        from astropy.cosmology import Planck18
        import sys
        sys.path.insert(0,str(ROOT.parent.parent/'GraphWeb_DESI/workflows/presentation'))
        from loa_tng_abacus_overlay_20261015 import unit,VAC
        a=np.load(HOME/'loa_tng_abacus_overlay_20261015/loa_predictions.npz')
        d=fitsio.read(VAC,columns=['TARGETID','RA','DEC','Z'])
        zg=np.linspace(.1,.6,10001);rg=Planck18.comoving_distance(zg).value
        xyz=unit(d['RA'],d['DEC'])*np.interp(d['Z'],zg,rg)[:,None]
        qx=xyz[a['catalogue_rows']];pad=64/Planck18.h
        keep=np.all((xyz>=qx.min(0)-pad)&(xyz<=qx.max(0)+pad),axis=1)
        ids=np.flatnonzero(keep);pos=xyz[ids];targetid=d['TARGETID'][ids]
        query=np.searchsorted(ids,a['catalogue_rows'])
        assert np.array_equal(targetid[query],a['TARGETID'])
        alpha=6.906260417222935
    print('TOPOLOGY',arm,len(pos),alpha,flush=True)
    st=gudhi.AlphaComplex(points=pos).create_simplex_tree(max_alpha_square=alpha**2)
    edges=[];tetra=[]
    for simplex,filtration in st.get_filtration():
        if len(simplex)==2:edges.append(simplex)
        elif len(simplex)==4:tetra.append(simplex)
    np.savez_compressed(OUT/f'{arm}_topology.npz',pos=pos,ids=ids,targetid=targetid,query=query,
        edges=np.asarray(edges,np.int32),tetra=np.asarray(tetra,np.int32),alpha=alpha)
    print('TOPOLOGY DONE',arm,len(edges),len(tetra),flush=True)

def metrics(arm):
    import networkx as nx
    d=np.load(OUT/f'{arm}_topology.npz');pos=d['pos'];e=d['edges'];n=len(pos)
    length=np.linalg.norm(pos[e[:,0]]-pos[e[:,1]],axis=1)
    row=np.concatenate([e[:,0],e[:,1]]);col=np.concatenate([e[:,1],e[:,0]])
    vals=np.tile(length,2);a=sparse.csr_matrix((vals,(row,col)),shape=(n,n));a.sort_indices()
    deg=np.diff(a.indptr);weighted=np.asarray(a.sum(1)).ravel()
    mn=np.full(n,np.inf);mx=np.zeros(n);np.minimum.at(mn,row,vals);np.maximum.at(mx,row,vals);mn[deg==0]=0
    mean=np.divide(weighted,deg,out=np.zeros(n),where=deg>0)
    b=a.copy();b.data=np.cbrt(b.data/b.data.max());clust=np.zeros(n)
    for lo in range(0,n,4096):
        hi=min(lo+4096,n);s=np.asarray((b[lo:hi]@b).multiply(b[lo:hi]).sum(1)).ravel()
        den=deg[lo:hi]*(deg[lo:hi]-1);clust[lo:hi]=np.divide(s,den,out=np.zeros_like(s),where=den>0)
    count=np.zeros(n);volume=np.zeros(n);tet=d['tetra']
    for lo in range(0,len(tet),50000):
        t=tet[lo:lo+50000];p=pos[t];v=np.abs(np.linalg.det(p[:,1:]-p[:,:1]))/6
        np.add.at(count,t.ravel(),1);np.add.at(volume,t.ravel(),np.repeat(v,4))
    density=np.divide(count,volume*np.maximum(deg,1),out=np.zeros(n),where=volume>0)
    adj=a.copy();adj.data[:]=1
    nd=np.divide(adj@density,deg,out=np.zeros(n),where=deg>0)
    centres=np.divide(adj@pos,deg[:,None],out=np.zeros((n,3)),where=deg[:,None]>0)
    cov=np.zeros((n,3,3))
    for i in range(3):
        for j in range(i,3):
            value=np.divide(adj@(pos[:,i]*pos[:,j]),deg,out=np.zeros(n),where=deg>0)-centres[:,i]*centres[:,j]
            cov[:,i,j]=value;cov[:,j,i]=value
    eig=np.maximum(np.linalg.eigvalsh(cov),0);eig[deg<3]=0
    x=np.column_stack([weighted,mean,mn,mx,clust,density,nd,eig])
    assert x.shape==(n,10) and np.isfinite(x).all() and (x>=0).all()
    # Independent exact NetworkX reference for the weighted-clustering contract.
    probes=np.linspace(0,n-1,24).astype(int)
    g=nx.Graph();g.add_nodes_from(range(n));g.add_weighted_edges_from((int(u),int(v),float(w)) for (u,v),w in zip(e,length))
    ref=np.array(list(nx.clustering(g,nodes=probes,weight='weight').values()))
    assert np.allclose(ref,clust[probes],atol=1e-12,rtol=1e-10)
    for i in probes:
        nei=a.indices[a.indptr[i]:a.indptr[i+1]]
        if len(nei)>=3:
            p=pos[nei]-pos[nei].mean(0)
            assert np.allclose(np.maximum(np.linalg.eigvalsh(p.T@p/len(p)),0),eig[i],atol=1e-8,rtol=1e-6)
    np.savez_compressed(OUT/f'{arm}_graph.npz',x=x,indptr=a.indptr,indices=a.indices,length=a.data.astype(np.float32),
        ids=d['ids'],query=d['query'],targetid=d['targetid'],pos=pos)
    dump(f'{arm}_metrics.json',{'nodes':n,'edges':len(e),'alpha_Mpc':float(d['alpha']),
        'weighted_clustering_max_error':float(np.max(abs(ref-clust[probes]))),'independent_probes':len(probes),
        'features':['weighted degree','mean length','min length','max length','weighted clustering','tetra density','neighbor tetra density','cov eig1','cov eig2','cov eig3']})
    print('METRICS DONE',arm,x.shape,flush=True)


def setup_torch():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    return torch


def blocks(adj,targets):
    """Exact incoming dependency blocks, including all neighbors at each hop."""
    import torch
    dst=np.asarray(targets);result=[]
    for _ in range(4):
        a=adj[dst].tocoo()
        src=np.concatenate([dst,np.setdiff1d(np.unique(a.col),dst)])
        lookup=np.full(adj.shape[0],-1,np.int64);lookup[src]=np.arange(len(src))
        ei=torch.from_numpy(np.stack([lookup[a.col],a.row]).astype(np.int64))
        result.append((ei,torch.from_numpy(a.data[:,None].copy()),len(dst)))
        dst=src
    return src,list(reversed(result))


def forward_blocks(model,x,bs):
    import torch.nn.functional as F
    for i,(edges,length,n) in enumerate(bs,1):
        residual=x[:n]
        x=getattr(model,f'gat_layer{i}')((x,residual),edges,edge_attr=length)
        if i<4:
            x=getattr(model,f'norm{i}')(x)
            if i>1:x=x+residual
            x=model.gat_dropout(F.relu(x))
    return x


def predict(model,x,adj,ids,batch=32):
    import torch
    model.eval();out=[]
    with torch.no_grad():
        for lo in range(0,len(ids),batch):
            src,bs=blocks(adj,ids[lo:lo+batch]);out.append(forward_blocks(model,x[src],bs).numpy())
    return np.concatenate(out)


def train(arm,radius,steps):
    torch=setup_torch()
    from sklearn.preprocessing import PowerTransformer
    from sklearn.metrics import confusion_matrix,balanced_accuracy_score,f1_score
    from gnn_models import SimpleGAT
    d=np.load(OUT/f'{arm}_graph.npz');ids=d['ids'];n=len(ids)
    adj=sparse.csr_matrix((d['length'],d['indices'],d['indptr']),shape=(n,n))
    labels=(np.load(OLD/'truth.npz')[f'eig_R{radius}_n512'][ids]>=.2).sum(1)
    f=np.load(OLD/'features.npz')
    trainids=np.flatnonzero(np.isin(ids,f['train_ids']))
    shared_validation=np.intersect1d(f['thin_20261015_tracer_ids'],f['validation_ids'])
    shared_test=np.intersect1d(f['thin_20261015_tracer_ids'],f['test_ids'])
    validids=np.flatnonzero(np.isin(ids,shared_validation))
    testids=np.flatnonzero(np.isin(ids,shared_test))
    rng=np.random.default_rng(SEED);torch.manual_seed(SEED)
    validids=rng.choice(validids,min(512,len(validids)),replace=False)
    testids=rng.choice(testids,min(2048,len(testids)),replace=False)
    scaler=PowerTransformer(method='box-cox').fit(d['x'][trainids]+1e-6)
    x=torch.from_numpy(scaler.transform(d['x']+1e-6).astype(np.float32));y=torch.from_numpy(labels).long()
    counts=np.bincount(labels[trainids],minlength=4);assert np.all(counts>0)
    weights=torch.tensor(len(trainids)/(4*counts),dtype=torch.float32)
    model=SimpleGAT(10,4,num_heads=4)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.7,patience=2,min_lr=.0005)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    name=f'R{radius}_{arm}';history=[];best=float('inf');beststep=0;start=time.monotonic()
    # Verify exact block inference against the unchanged canonical forward.
    model.eval();probe=validids[:4];src,bs=blocks(adj,probe)
    sub=adj[src][:,src].tocoo();edge=torch.from_numpy(np.stack([sub.col,sub.row]).astype(np.int64))
    with torch.no_grad():
        direct=model(x[src],edge,torch.from_numpy(sub.data[:,None].copy()))[:len(probe)]
        blocked=forward_blocks(model,x[src],bs)
    err=float(torch.max(abs(direct-blocked)));assert err<2e-5,err
    print('BLOCK PARITY',name,err,'train',len(trainids),counts.tolist(),flush=True)
    for step in range(1,steps+1):
        model.train();target=rng.choice(trainids,16,replace=False);src,bs=blocks(adj,target)
        optimizer.zero_grad();logits=forward_blocks(model,x[src],bs);loss=criterion(logits,y[target]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%100==0 or step==steps:
            pred=predict(model,x,adj,validids);vl=float(criterion(torch.from_numpy(pred),y[validids]))
            ba=float(balanced_accuracy_score(labels[validids],pred.argmax(1)))
            history.append({'step':step,'train_loss':float(loss.detach()),'validation_loss':vl,'validation_balanced_accuracy':ba,'seconds':time.monotonic()-start})
            if vl<best:
                best=vl;beststep=step
                torch.save({'state':model.state_dict(),'scaler':scaler,'radius':radius,'arm':arm,'step':step,'seed':SEED,
                    'model':'canonical SimpleGAT(10,4,num_heads=4)','class_counts':counts,'block_parity_max_error':err},OUT/f'{name}.pt')
            if step>1:scheduler.step(vl)
            dump(f'{name}_history.json',history)
            print(name,history[-1],'best',beststep,flush=True)
    ck=torch.load(OUT/f'{name}.pt',weights_only=False);model.load_state_dict(ck['state'])
    pred=predict(model,x,adj,testids);prob=torch.softmax(torch.from_numpy(pred),1).numpy()
    np.savez_compressed(OUT/f'{name}_heldout.npz',ids=ids[testids],truth=labels[testids],probability=prob)
    dump(f'{name}_training.json',{'steps':steps,'best_step':beststep,'train_rows':len(trainids),'validation_rows':len(validids),'test_rows':len(testids),
        'balanced_accuracy':float(balanced_accuracy_score(labels[testids],pred.argmax(1))),
        'macro_f1':float(f1_score(labels[testids],pred.argmax(1),average='macro')),
        'confusion_matrix':confusion_matrix(labels[testids],pred.argmax(1),labels=np.arange(4)).tolist(),
        'seconds':time.monotonic()-start,'block_parity':err,
        'recipe':'Exact legacy architecture and full incoming context; 16 random target nodes per update, float32 Adam, train-only BoxCox and class weights, spatial label splits. One seed. New optimization experiment, not bitwise historical training replay.'})


def infer(arm,radius,steps=0):
    torch=setup_torch()
    from gnn_models import SimpleGAT
    from sklearn.preprocessing import PowerTransformer
    d=np.load(OUT/'loa_graph.npz');n=len(d['x']);adj=sparse.csr_matrix((d['length'],d['indices'],d['indptr']),shape=(n,n))
    ck=torch.load(OUT/f'R{radius}_{arm}.pt',weights_only=False)
    model=SimpleGAT(10,4,num_heads=4);model.load_state_dict(ck['state']);model.eval()
    historical=PowerTransformer(method='box-cox').fit_transform(d['x']+1e-6);historical-=historical.mean(0)
    output={'TARGETID':d['targetid'][d['query']]}
    # Exact layer-wise graph evaluation, bounded to destination batches.
    for convention,values in [('legacy_targetfit',historical),('frozen',ck['scaler'].transform(d['x']+1e-6))]:
        x=torch.from_numpy(values.astype(np.float32))
        with torch.no_grad():
            for layer in range(1,5):
                result=[]
                for lo in range(0,n,2048):
                    dst=np.arange(lo,min(lo+2048,n));a=adj[dst].tocoo()
                    src=np.concatenate([dst,np.setdiff1d(np.unique(a.col),dst)])
                    lookup=np.full(n,-1,np.int64);lookup[src]=np.arange(len(src))
                    ei=torch.from_numpy(np.stack([lookup[a.col],a.row]).astype(np.int64))
                    z=getattr(model,f'gat_layer{layer}')((x[src],x[dst]),ei,edge_attr=torch.from_numpy(a.data[:,None].copy()))
                    if layer<4:
                        z=getattr(model,f'norm{layer}')(z)
                        if layer>1:z=z+x[dst]
                        z=torch.relu(z)
                    result.append(z)
                x=torch.cat(result)
                print('INFER',arm,radius,convention,layer,flush=True)
        output[convention]=torch.softmax(x[d['query']],1).numpy()
        probes=d['query'][np.linspace(0,len(d['query'])-1,8).astype(int)]
        direct=predict(model,torch.from_numpy(values.astype(np.float32)),adj,probes,batch=8)
        err=float(np.max(abs(direct-x[probes].numpy())))
        assert err<2e-4,err
        output[convention+'_layerwise_parity']=np.array(err)
        print('INFERENCE PARITY',convention,err,flush=True)
    np.savez_compressed(OUT/f'R{radius}_{arm}_loa.npz',**output)


def verify(arm='dense'):
    torch=setup_torch()
    from sklearn.preprocessing import PowerTransformer
    from gnn_models import SimpleGAT
    import hashlib
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    expected={v['name']:v['sha256'] for v in json.loads((OLD/'MANIFEST.json').read_text())['files']}
    for name in ['catalogue.npz','features.npz','truth.npz']:assert sha(OLD/name)==expected[name]
    path=Path('/pscratch/sd/d/dkololgi/tng_illustris/cache/from_repo_root/processed_gcn_data_alpha_mc1e+09.pt')
    old,_,_=torch.load(path,map_location='cpu',weights_only=False)
    d=np.load(OUT/'dense_graph.npz');n=len(d['x']);assert len(old.x)==n
    poserr=float(np.max(abs(old.pos.numpy()-d['pos'])));assert poserr<1e-4
    a=sparse.csr_matrix((np.ones(len(d['indices'])),d['indices'],d['indptr']),shape=(n,n))
    oi=old.edge_index.numpy();b=sparse.csr_matrix((np.ones(oi.shape[1]),oi),shape=(n,n))
    edge_difference=(a!=b).nnz;assert edge_difference/max(a.nnz,b.nnz)<.001
    scaled=PowerTransformer(method='box-cox').fit_transform(d['x']+1e-6)
    diff=abs(scaled-old.x.numpy())
    report={'original_cache_sha256':sha(path),'dense_position_max_Mpc':poserr,
      'dense_directed_edge_symmetric_difference':edge_difference,'dense_directed_edges':a.nnz,
      'full_fit_scaled_feature_absolute_difference_quantiles':np.quantile(diff,[.5,.95,.99,1]).tolist(),
      'feature_absolute_difference_max_by_column':diff.max(0).tolist(),'models':{}}
    del old,scaled,diff,a,b
    reference_ids=None
    for arm in ['dense','thin']:
        d=np.load(OUT/f'{arm}_graph.npz');n=len(d['x'])
        adj=sparse.csr_matrix((d['length'],d['indices'],d['indptr']),shape=(n,n))
        for radius in [2,7]:
            name=f'R{radius}_{arm}';held=np.load(OUT/f'{name}_heldout.npz')
            if reference_ids is None:reference_ids=held['ids']
            assert np.array_equal(held['ids'],reference_ids)
            ck=torch.load(OUT/f'{name}.pt',weights_only=False)
            model=SimpleGAT(10,4,num_heads=4);model.load_state_dict(ck['state']);model.eval()
            x=torch.from_numpy(ck['scaler'].transform(d['x']+1e-6).astype(np.float32))
            probes=np.searchsorted(d['ids'],held['ids'][:8]);src,bs=blocks(adj,probes)
            sub=adj[src][:,src].tocoo();edge=torch.from_numpy(np.stack([sub.col,sub.row]).astype(np.int64))
            with torch.no_grad():
                ref=model(x[src],edge,torch.from_numpy(sub.data[:,None].copy()))[:len(probes)]
                exact=forward_blocks(model,x[src],bs)
            error=float(torch.max(abs(ref-exact)));assert error<2e-4
            assert np.allclose(torch.softmax(exact,1).numpy(),held['probability'][:8],atol=2e-5)
            labels=(np.load(OLD/'truth.npz')[f'eig_R{radius}_n512'][held['ids']]>=.2).sum(1)
            assert np.array_equal(labels,held['truth'])
            observed=np.load(OUT/f'{name}_loa.npz')
            report['models'][name]={'checkpoint_sha256':sha(OUT/f'{name}.pt'),
                'trained_exact_block_parity':error,'heldout_same_ids':True,
                'loa_inference_parity':{k:float(observed[k+'_layerwise_parity']) for k in ['legacy_targetfit','frozen']}}
    report['source_sha256']=sha(__file__);report['verified']=True
    dump('VALIDATION.json',report)
    dump('MANIFEST.json',{'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='MANIFEST.json']})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['topology','metrics','train','infer','verify']);p.add_argument('arm',choices=['dense','thin','loa']);p.add_argument('--radius',type=int,choices=[2,7],default=2);p.add_argument('--steps',type=int,default=1200);args=p.parse_args()
    if args.stage in ['train','infer']:globals()[args.stage](args.arm,args.radius,args.steps)
    else:globals()[args.stage](args.arm)
