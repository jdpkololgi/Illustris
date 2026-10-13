"""Real-shape engineering checks; no scientific fit or calibration claim."""
import copy
import json
import time
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi.e2e_wholefield_pilot_data import ROOT, SHAPE
from workflows.sbi.e2e_wholefield_pilot_models import FineModel, Condition, sample
from workflows.sbi.e2e_wholefield_pilot_train import setup, panel, batch, update, sources
from workflows.sbi.e2e_coupled_cfm_pilot import save
from workflows.sbi.e2e_vdm_context_models import block_mean, decode_density


def run():
    setup(17)
    root = ROOT/'smoke'; root.mkdir(exist_ok=True)
    chart = views.load_chart(c.sha256(coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'))
    addresses = panel(['ph000'])[:2]; x, cond = batch(addresses, chart)
    weight = torch.ones(SHAPE, device='cuda')
    rows = []
    for architecture in ('unet', 'transformer'):
        torch.cuda.reset_peak_memory_stats()
        model = FineModel(architecture, 'v').cuda()
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        start = time.monotonic()
        update(model, ema, optimizer, weight, x, cond, 17, 0)
        path = root/f'{architecture}_restart.pt'
        save(path, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict()))
        expected_row = update(model, ema, optimizer, weight, x, cond, 17, 1)
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saved = torch.load(path, map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        actual_row = update(model, ema, optimizer, weight, x, cond, 17, 1)
        maximum = max(float((v.cpu()-expected[k]).abs().max()) for k, v in model.state_dict().items())
        if maximum > 2e-6 or abs(actual_row['loss']-expected_row['loss']) > 1e-6:
            raise ValueError('Restart replay exceeds numerical tolerance')
        evaluation = Condition(cond.joint[:1], cond.wide[:1], cond.coarse_wide[:1], 'oracle_diagnostic')
        noise = torch.randn_like(x[:1])
        value = sample(model, evaluation, noise, nfe=4)
        cs = chart['coarse_logrho']; mass = (evaluation.coarse_joint().double()*cs['std'][0]+cs['mean'][0]).exp()
        density = decode_density(block_mean(mass), value.double()*chart['fine_residual']['std'][0])
        relative = float(((block_mean(density)-block_mean(mass))/block_mean(mass)).abs().max())
        if not torch.isfinite(density).all() or density.min() <= 0 or relative > 1e-9:
            raise ValueError('Decoder positivity/mass check failed')
        # Clean prediction has its own objective; do not reinterpret a velocity
        # fit as a calibrated endpoint model. Exercise its backward path only.
        model.parameterization = 'x'
        endpoint_row = update(model, ema, optimizer, weight, x, cond, 17, 2)
        rows.append(dict(architecture=architecture, seconds=time.monotonic()-start,
                         restart_parameter_maxabs=maximum, mass_relative=relative,
                         peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                         parameters=sum(p.numel() for p in model.parameters())))
        print(json.dumps(rows[-1]), flush=True)
        del model, ema, optimizer, saved, expected, value, density
        torch.cuda.empty_cache()
    c.atomic_json(root/'COMPLETE.json', dict(passed=True, sources=sources(), rows=rows,
        addresses=addresses, scope='engineering only, not posterior calibration'))


if __name__ == '__main__':
    run()
