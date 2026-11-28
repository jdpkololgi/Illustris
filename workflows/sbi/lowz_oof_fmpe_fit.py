"""One 16-rank FMPE estimator; global batch 4096, no development data used."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from sbi.neural_nets import posterior_flow_nn
from workflows.sbi.lowz_development_table import sha256

class Objective(torch.nn.Module):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self,y,x):
        return self.estimator.loss(y,x).mean()

def main(seed):
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16
    dest = Path(os.environ["FMPE_RUN"])
    marker = dest/"P12A_DATASET_READY.json"
    ready = json.loads(marker.read_text())
    assert ready["pass"] and not ready["sealed_phase_opened"]
    ckdir = dest / f"seed_{seed}"
    ckdir.mkdir(exist_ok=True)
    with np.load(ready["training"]["path"]) as f:
        raw = [f[k] for k in ("context","theta_softplus")]
    stats = [(a.mean(0,dtype="f8"),a.std(0,dtype="f8")) for a in raw]
    assert all(np.all(s>0) for _,s in stats)
    x,y = [torch.from_numpy(((a-m)/s).astype("f4")).cuda() for a,(m,s) in zip(raw,stats)]
    torch.manual_seed(seed)
    estimator = posterior_flow_nn(model="mlp",hidden_features=128,num_layers=5,
        z_score_theta="none",z_score_x="none")(y[:4096].cpu(),x[:4096].cpu()).cuda()
    net = DistributedDataParallel(Objective(estimator),device_ids=[device])
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-4)
    # Identical internal holdback for both estimators; no development labels accessed.
    g = torch.Generator(device="cuda").manual_seed(12142)
    order = torch.randperm(len(x),device="cuda",generator=g)
    validation,train = order[:len(x)//10],order[len(x)//10:]
    torch.manual_seed(seed+rank+1)
    best,stale = float("inf"),0
    lo,hi = y.amin(0),y.amax(0)
    for epoch in range(1,301):
        net.train()
        g.manual_seed(seed*1000+epoch)
        indices = train[torch.randperm(len(train),device="cuda",generator=g)]
        batches = indices[:len(indices)//4096*4096].view(-1,world,4096//world)[:,rank]
        for b in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = net(y[b],x[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            metric = torch.zeros((),device="cuda",dtype=torch.float64)
            for b in validation[rank::world].split(1024):
                metric += estimator.loss(y[b],x[b]).sum()
            dist.all_reduce(metric)
            val = metric.item()/len(validation)
        if not np.isfinite(val):
            raise RuntimeError("nonfinite validation loss")
        if val < best:
            best,stale = val,0
            if rank == 0:
                tmp = ckdir/"best.tmp"
                torch.save(dict(schema_version="p12a-fmpe-estimator-v1",state_dict=estimator.state_dict(),
                    context_mean=stats[0][0],context_std=stats[0][1],theta_mean=stats[1][0],theta_std=stats[1][1],
                    prior_low=(lo-.2*(hi-lo)).cpu().numpy(),prior_high=(hi+.2*(hi-lo)).cpu().numpy(),
                    hidden_features=128,num_layers=5,seed=seed,epoch=epoch,validation_loss=val,
                    dataset_marker_sha256=sha256(marker)),tmp)
                tmp.replace(ckdir/"fmpe_estimator.pt")
        else:
            stale += 1
        if rank == 0:
            print(f"epoch={epoch} validation_loss={val:.6g} best={best:.6g}",flush=True)
        if stale >= 20:
            break
    if rank == 0:
        ck = ckdir/"fmpe_estimator.pt"
        restored = torch.load(ck,map_location="cpu",weights_only=False)
        estimator.load_state_dict(restored["state_dict"],strict=True)
        (ckdir/"FIT_COMPLETE.json").write_text(json.dumps(dict(seed=seed,epochs=epoch,best_epoch=restored["epoch"],
            checkpoint_sha256=sha256(ck),development_calibrated=False)))
    dist.destroy_process_group()

if __name__ == "__main__":
    main(int(sys.argv[1]))
