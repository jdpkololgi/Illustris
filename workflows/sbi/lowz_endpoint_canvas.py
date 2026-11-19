"""Fine shared-state Heun integration for the expanded low-z endpoint U-Net.

One cap and draw per canvas. Caller supplies observation-only conditions
with crops of the same sampled wide field. No truth readers or scheduling.
"""
import numpy as np
import torch
from workflows.sbi.e2e_global_canvas import Geometry
from workflows.sbi.e2e_vdm_context_models import project


def geometry(midpoints):
    mid = np.asarray(midpoints)
    if (mid.ndim != 2 or mid.shape[1] != 3 or not len(mid)
            or not np.issubdtype(mid.dtype, np.integer) or np.any(mid % 8)
            or len(np.unique(mid, axis=0)) != len(mid)):
        raise ValueError("Expected unique raw-grid midpoints aligned to coarse cells")
    window = np.array([96, 64, 64])
    starts = (mid - window) // 2
    origin = starts.min(0)
    relative = starts - origin
    return Geometry(tuple(map(tuple, relative)), tuple(relative.max(0)+window),
                    tuple(origin), tuple(window))


@torch.inference_mode()
def integrate(model, conditions, noise, geom, nfe):
    if (nfe < 2 or nfe % 2 or tuple(noise.shape) != (1, 1, *geom.shape)
            or tuple(geom.window) != (96, 64, 64) or len(conditions) != len(geom.starts)
            or model.parameterization != 'x'):
        raise ValueError("Wrong endpoint canvas configuration")
    model.eval()
    state = project(noise.float())
    weights = geom.weights(state.device)
    den = weights.clamp_min(1)
    # Uncovered cells are not posterior draws: return coverage explicitly.
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=state.is_cuda):
        def velocity(z, t):
            total = torch.zeros_like(z)
            for i, condition in enumerate(conditions):
                sl = geom.crop(i)
                local = z[sl]
                total[sl] += model.velocity(local, local.new_full((1,), t),
                                            model.prepare(condition))
            return total / den

        dt = 2 / nfe
        for i in range(nfe//2):
            t = i * dt
            a = velocity(state, t)
            b = velocity(project(state + dt*a), t+dt)
            state = project(state + .5*dt*(a+b))
    if not torch.isfinite(state).all():
        raise FloatingPointError("Nonfinite fine canvas")
    return state, weights
