"""Full-size synthetic CPU connectivity smoke; not GPU timing or model fitting."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_benchmark_models as models
from workflows.sbi.e2e_vdm_context_models import project,block_mean


def run():
    c.require_compute(); coord.require_host_checks(); torch.set_num_threads(4)
    for item in json.loads((c.REPO/'SOURCE.json').read_text())['files']:
        if c.sha256(c.REPO/item['relative'])!=item['sha256']:
            raise ValueError('full-size CPU smoke source drift')
    torch.manual_seed(20260918)
    reports=[]
    for stage,domain,region in [('coarse','wide','wide'),('fine','parent','left'),
                                ('fine','parent','right'),('fine','joint','joint')]:
        model=models.CoupledBackbone(stage,domain).cpu()
        # The production constructor's zero final head initially blocks upstream
        # gradients. Perturb it only for this explicit connectivity probe; no
        # optimizer update or scientific target is involved.
        torch.nn.init.normal_(model.output.weight,std=.001)
        for kind in ('vdm','cfm'):
            model.zero_grad(set_to_none=True)
            joint=torch.randn(1,12,64,48,48,requires_grad=True)
            wide=torch.randn(1,12,48,48,48,requires_grad=True)
            offset=torch.tensor([[108.256,0.,0.]])
            cj=torch.zeros(1,1,64,48,48) if stage=='fine' else None
            cw=torch.zeros(1,1,48,48,48) if stage=='fine' else None
            condition=models.FieldCondition(joint,wide,offset,region,cj,cw,
                                             'training_truth' if stage=='fine' else None)
            target=torch.randn(1,1,*models.SHAPES[region])
            if stage=='fine': target=project(target)
            started=time.monotonic()
            value,parts=models.objective(model,target,condition,torch.Generator().manual_seed(71),kind)
            if not torch.isfinite(value): raise FloatingPointError('nonfinite full-size CPU objective')
            value.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('nonfinite full-size parameter gradient')
            if joint.grad is None or wide.grad is None:
                raise ValueError('observation encoder disconnected')
            # This rejects dependence only on a global joint-field summary:
            # a mean-only pathway has constant gradient within each channel.
            spatial_joint=joint.grad-joint.grad.mean((2,3,4),keepdim=True)
            joint_spatial_l1=float(spatial_joint.abs().sum())
            wide_l1=float(wide.grad.abs().sum())
            if joint_spatial_l1<=0 or wide_l1<=0:
                raise ValueError('spatial joint/wide observation pathway is absent')
            outside=None
            if region in ('left','right'):
                outside=float((joint.grad[:,:,48:] if region=='left' else joint.grad[:,:,:16]).abs().sum())
                if outside<=0: raise ValueError('independent parent lost full joint observations')
            with torch.no_grad(): output=model(target,model.schedule(torch.tensor([.5])),condition)
            zero=float(block_mean(output).abs().max()) if stage=='fine' else None
            if not torch.isfinite(output).all() or (zero is not None and zero>2e-6):
                raise ValueError('full-size output/projector numerical failure')
            result=dict(stage=stage,domain=domain,region=region,objective=kind,
                shape=list(target.shape),parameters=sum(p.numel() for p in model.parameters()),
                loss=float(value.detach()),independent_dimensions=parts['independent_dimensions'],
                spatial_joint_gradient_l1=joint_spatial_l1,wide_gradient_l1=wide_l1,
                outside_parent_joint_gradient_l1=outside,max_output_block_mean=zero,
                elapsed_cpu_seconds=time.monotonic()-started)
            reports.append(result); print(json.dumps(result),flush=True)
            del joint,wide,condition,target,value,parts,output,spatial_joint,cj,cw
        del model; gc.collect()
    if len({row['parameters'] for row in reports})!=1:
        raise ValueError('matched coupled backbones differ in parameter count')
    result=dict(**coord.provenance(),reports=reports,synthetic_only=True,optimizer_updates=0,
        final_head_perturbed_for_connectivity_probe=True,gpu_timing=False,
        scientific_training_authorized=False,posterior_performance_evaluated=False,
        source_code_sha256=c.sha256(__file__),model_code_sha256=c.sha256(models.__file__),
        outputs=[],**{'pass':True})
    path=coord.ROOT/'technical_cpu'/f'MODEL_INTERFACE_SMOKE_{time.time_ns()}.json'
    c.atomic_json(path,result)
    print(json.dumps(dict(path=str(path),sha256=c.sha256(path),cases=len(reports))),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); run()
