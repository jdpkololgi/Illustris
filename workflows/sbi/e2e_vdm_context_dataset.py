"""Role-checked model inputs with strict observation/target access separation."""
from pathlib import Path
import h5py
import numpy as np
import torch

from workflows.sbi.e2e_vdm_context_data import ROLES, CONFIG, spec, read_json, guarded, output_root
from workflows.sbi.e2e_vdm_context_products import pair_rows
from workflows.sbi.e2e_field_build_products import sha256, require_compute, eigs
from workflows.sbi.e2e_field_dataset import Moments
from workflows.sbi.e2e_durable import publish_json
from workflows.sbi.e2e_vdm_context_models import FineCondition, encode_density, replicate


def context_crop(offset_raw):
    offset = np.asarray(offset_raw, dtype=int)
    if offset.shape != (3,) or any(offset % 8) or np.max(np.abs(offset)) > 32:
        raise ValueError('unaligned/out-of-range context offset')
    return tuple(slice(4+int(o)//8, 52+int(o)//8) for o in offset)


def coarse_local_crop(core_offset_raw, context_offset_raw):
    relative = np.asarray(core_offset_raw)-np.asarray(context_offset_raw)
    if relative.shape != (3,) or any(relative % 8):
        raise ValueError('coarse and fine lattices not aligned')
    start = 18+relative//8
    if np.any(start<0) or np.any(start+12>48):
        raise ValueError('fine parent outside shared coarse domain')
    return tuple(slice(int(i),int(i+12)) for i in start)


def observation_summary(raw):
    core=(slice(16,32),)*3
    x=raw['local'][(slice(None),*core)]
    return dict(mean_log1p_count=float(x[0].mean()),support_fraction=float(raw['support'][core].mean()),
        angular_response=float(x[2].mean()),boundary_distance_mpc=float(x[6].mean()),
        mean_log_nbar=float(x[7].mean()),observer_redshift=float(x[11].mean()))


class Products:
    def __init__(self, root, phases, *, targets=False, confirmation_receipt=None, verify=True):
        self.root, self.c = output_root(root), spec()
        self.phases, self.targets = tuple(phases), targets
        if not set(phases) <= ROLES.keys() or len(set(phases)) != len(phases):
            raise PermissionError('unregistered phase access')
        if targets and 'ph005' in phases:
            if confirmation_receipt is None:
                raise PermissionError('confirmation scores require frozen full-matrix receipt')
            release = read_json(confirmation_receipt)
            if release.get('all_models_frozen') is not True or release.get('geometry_sha256') != sha256(self.root/'data/GEOMETRY.json'):
                raise PermissionError('invalid confirmation release')
        geometry = read_json(self.root/'data/GEOMETRY.json')
        if not geometry['quotas_pass'] or geometry['config_sha256'] != sha256(CONFIG):
            raise ValueError('geometry drift/missing gate')
        companions, self.pairs = pair_rows(geometry['rows'])
        self.rows = {r['anchor_id']:r for r in geometry['rows']+companions if r['phase'] in phases}
        self.receipts = {}
        for phase in phases:
            path = self.root/'data'/phase/'COMPLETE.json'
            receipt = read_json(path)
            if (receipt['phase'] != phase or receipt['role'] != ROLES[phase]
                    or receipt['binding']['config_sha256'] != sha256(CONFIG)
                    or receipt['binding']['geometry_sha256'] != sha256(self.root/'data/GEOMETRY.json')):
                raise ValueError('phase product binding mismatch')
            self.receipts[phase] = sha256(path)
            if verify:
                require_compute()
                for file in receipt['files']:
                    p = guarded(file['path'],phase)
                    if p.name == 'targets.h5' and not targets:
                        continue
                    if sha256(p) != file['sha256']:
                        raise ValueError('product payload drift')
        self.chart = None
        chart_path = self.root/'data/NORMALIZATION.json'
        if chart_path.exists():
            self.chart = read_json(chart_path)
            if self.chart['config_sha256'] != sha256(CONFIG) or self.chart['geometry_sha256'] != sha256(self.root/'data/GEOMETRY.json'):
                raise ValueError('normalization binding drift')

    def raw_observations(self, anchor, offset_raw=(0,0,0)):
        require_compute()
        row = self.rows[anchor]
        path = self.root/'data'/row['phase']/f"{row['cap']}_observations.h5"
        with h5py.File(guarded(path,row['phase']),'r') as f:
            if f.attrs['contains_matter'] or f.attrs['role'] != ROLES[row['phase']]:
                raise PermissionError('not an observation-only product')
            group = f[anchor]
            local = group['local'][:]
            wide = group['wide_extended'][(slice(None),*context_crop(offset_raw))]
            support = group['support'][:]
        return dict(local=local,wide=wide,support=support)

    def raw_targets(self, anchor, offset_raw=(0,0,0)):
        require_compute()
        if not self.targets:
            raise PermissionError('observation-only reader cannot open target files')
        row = self.rows[anchor]
        domain = row.get('parent_domain_anchor',anchor)
        with h5py.File(guarded(self.root/'data'/row['phase']/'targets.h5',row['phase']),'r') as f:
            if f.attrs['contains_observations'] or f.attrs['role'] != ROLES[row['phase']]:
                raise PermissionError('target role mismatch')
            rho = f[anchor]['rho_parent'][:]
            coarse = f[domain]['coarse_rho_extended'][context_crop(offset_raw)]
            tensor = f[anchor]['fullbox_tensor_core'][:]
        return dict(rho=rho,coarse=coarse,tensor=tensor)

    def condition(self, anchor, offset_raw=(0,0,0), *, device='cpu', coarse=None, coarse_source=None):
        """Only observational reads; `coarse` is a caller-supplied physical draw."""
        if self.chart is None:
            raise ValueError('frozen train-only normalization required')
        raw = self.raw_observations(anchor,offset_raw)
        n = self.chart
        local = (raw['local']-np.array(n['local']['mean'])[:,None,None,None])/np.array(n['local']['std'])[:,None,None,None]
        summary = raw['wide'].mean((1,2,3),dtype=np.float64)
        summary = (summary-np.array(n['summary']['mean']))/np.array(n['summary']['std'])
        local = np.concatenate([local,np.broadcast_to(summary[:,None,None,None],(12,48,48,48))])
        wide = (raw['wide']-np.array(n['wide']['mean'])[:,None,None,None])/np.array(n['wide']['std'])[:,None,None,None]
        tensor = lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device).unsqueeze(0)
        relative = np.asarray(self.rows[anchor].get('core_offset_raw',[0,0,0]))-np.asarray(offset_raw)
        offset = tensor(relative*self.c['raw_cell_mpc']*self.c['coordinate_h'])
        local_coarse, wide_coarse = None, None
        if coarse is not None:
            coarse = torch.as_tensor(coarse,device=device,dtype=torch.float32)
            if coarse.shape != (1,1,48,48,48) or not torch.isfinite(coarse).all() or (coarse <= 0).any():
                raise ValueError('one positive physical coarse draw required')
            if coarse_source not in ('sampled','fixed_mean','oracle_diagnostic','training_truth'):
                raise PermissionError('explicit coarse provenance required')
            stats = n['coarse']
            wide_coarse = (coarse.log()-stats['mean'])/stats['std']
            sl = coarse_local_crop(self.rows[anchor].get('core_offset_raw',[0,0,0]),offset_raw)
            local_coarse = replicate(wide_coarse[(slice(None),slice(None),*sl)],4)
        elif coarse_source is not None:
            raise ValueError('coarse source without a field')
        return FineCondition(tensor(local),tensor(wide),offset,local_coarse,wide_coarse,coarse_source)


def fit_normalization(root):
    require_compute()
    root = output_root(root)
    dataset = Products(root,['ph000','ph002'],targets=True)
    if (root/'data/NORMALIZATION.json').exists():
        raise FileExistsError('normalization already frozen')
    ids = sorted(k for k,r in dataset.rows.items() if r['small_train'])
    if len(ids) != 32:
        raise ValueError('exactly A32 normalization fields required')
    local, wide, summary = ([Moments() for _ in range(12)] for _ in range(3))
    fine, coarse, residual = Moments(), Moments(), Moments()
    physical = [Moments() for _ in range(6)]
    descriptors=[]
    for anchor in ids:
        target = dataset.raw_targets(anchor)
        rho = torch.from_numpy(target['rho'])[None,None]
        _, u = encode_density(rho)
        fine.add(np.log(target['rho']))
        residual.add(u.numpy())
        eigen = eigs(target['tensor'])
        features = np.concatenate([(target['rho'][16:32,16:32,16:32]-1)[...,None],
                                   eigen,np.diff(eigen,axis=-1)],axis=-1)
        for channel,moment in enumerate(physical):
            moment.add(features[...,channel])
        for offset_index,offset in enumerate(dataset.c['context_offsets_raw']):
            observation = dataset.raw_observations(anchor,offset)
            if offset_index==0:
                descriptors.append(observation_summary(observation))
            target = dataset.raw_targets(anchor,offset)
            coarse.add(np.log(target['coarse']))
            for channel in range(12):
                if offset_index == 0:
                    local[channel].add(observation['local'][channel])
                wide[channel].add(observation['wide'][channel])
                summary[channel].add(observation['wide'][channel].mean(dtype=np.float64))
    channels = lambda moments: {key:[m.report()[key] for m in moments] for key in ('mean','std')}
    residual_report = residual.report()
    if abs(residual_report['mean']) > 1e-12:
        raise ValueError('residual chart mean not zero')
    result = dict(config_sha256=sha256(CONFIG),geometry_sha256=sha256(root/'data/GEOMETRY.json'),
        fit_ids=ids,fit_phases=['ph000','ph002'],product_receipts=dataset.receipts,
        local=channels(local),wide=channels(wide),summary=channels(summary),
        fine=fine.report(),coarse=coarse.report(),residual=dict(residual_report,mean=0.),
        physical_probe_scales=dict(channels(physical),names=['delta','lambda1','lambda2','lambda3','gap12','gap23']),
        conditioning_bin_edges={key:np.quantile([d[key] for d in descriptors],[1/3,2/3]).tolist()
                                for key in descriptors[0]},
        context_offsets_fit=dataset.c['context_offsets_raw'],per_patch_mean_removed=False)
    publish_json(root/'data/NORMALIZATION.json',result)
    return result
