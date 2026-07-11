#!/usr/bin/env python3
"""Gate F3 — generative (calibrated) F-tier via ENERGY-SCORE training.

docs/plan_field_level_multimodal.md §3 (F3 row) + §12 (v2). Turns the F-tier
from a POINT estimate into a CALIBRATED sample-based posterior by making the
decoded density field delta_hat STOCHASTIC (latent z ~ N(0,I) injected into the
field decoder) and propagating samples through the FIXED FFT physics layer +
analytic Cardano eigensolver. Each z -> a different delta_hat -> (physics) ->
one eigenvalue sample per galaxy. The encoder stays DETERMINISTIC; only the
field decode is stochastic (branch-(a) §11 showed the info F-tier needs lives in
the post-physics field, not the encoder embedding h_i).

Base = config A (gate_ftier_v2 winner): union graph + attention-aggregation
encoder + union edge_attr features + TSC scatter + U-Net decoder + FFT physics +
Cardano eigvalsh3x3. All of that is REUSED unchanged from
gate_ftier_v2.py / gate_t4_graph_field_poisson.py.

Training objective — the ENERGY SCORE (strictly proper, likelihood-free;
Gneiting & Raftery 2007; Pacchiardi & Dutta 2022). Per step draw M z-samples,
push each through decoder+physics -> {lambda_m} per galaxy, with truth y (all in
STANDARDIZED-eigenvalue space):

  ES_i = mean_m ||lambda_m,i - y_i||_2  -  0.5 * mean_{m!=m'} ||lambda_m,i - lambda_m',i||_2

averaged over TRAIN galaxies; minimise ES. The repulsion term (-0.5 * ...) makes
this proper and resists posterior collapse. Gradients flow through the physics
layer as usual.

Eval: draw K samples per galaxy on the TEST split (transductive: K global-grid
forwards, each yields lambda for ALL galaxies) -> posterior. Report posterior-mean
lambda1/2/3 R^2 (expect the ~0.84 F-tier ceiling), cluster-slice Spearman,
SBC KS-uniform p per eigenvalue, and central lambda1 coverage @68/90% (same
calibration machinery as gate_g6_fmpe_frozen_head.py). Sanity: every sample is
ascending by construction (eigvalsh3x3).

Honest framing: F-tier is accuracy-capped ~0.84 (§12 negative result), so this
will NOT beat G3+FMPE on eigenvalue accuracy (P3: FMPE lambda1 R^2 0.850,
cov 0.594@68 pre-tempering). The question F3 answers is whether a PRINCIPLED
generative F-tier gives a CALIBRATED posterior.

Torch; GPU required for the real run (fail-fast assert), CPU only for --smoke.
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kstest
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# reuse validated components (config A): attention encoder + TSC geometry from v2,
# physics layer + Cardano eigensolver + RSMOOTH from v1.
from gate_ftier_v2 import EGNNAttnEncoder, Geometry, zscore, RA_MIN, RA_MAX, DEC_MIN, DEC_MAX
from gate_t4_graph_field_poisson import PhysicsLayer, eigvalsh3x3, UNet3D, RSMOOTH_MPC


# ---------------------------------------------------------------- stochastic decoder
class FiLMUNet3D(nn.Module):
    """Config-A U-Net (same topology as the validated gate_t4 UNet3D) but each
    encoder/mid/decoder block is FiLM-modulated (per-channel scale/shift) by a
    latent z ~ N(0,I). A shared MLP maps z -> (gamma, beta) for every block, so a
    single deterministic scattered grid yields a DIFFERENT delta_hat per z. FiLM
    (multiplicative + additive) gives z strong, expressive control over the field,
    which resists the energy-score posterior-collapse failure mode a purely
    additive broadcast channel is prone to.
    """

    def __init__(self, c_in, zdim, base=16):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.SiLU(),
                                 nn.Conv3d(co, co, 3, padding=1), nn.SiLU())

        self.e0 = blk(c_in, base)
        self.e1 = blk(base, base * 2)
        self.pool = nn.AvgPool3d(2)
        self.mid = blk(base * 2, base * 2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.d1 = blk(base * 4, base * 2)
        self.d0 = blk(base * 3, base)
        self.out = nn.Conv3d(base, 1, 1)
        # FiLM channel widths for e0, e1, mid, d1, d0 (block outputs)
        self._film_ch = [base, base * 2, base * 2, base * 2, base]
        tot = sum(self._film_ch)
        self.film = nn.Sequential(nn.Linear(zdim, 64), nn.SiLU(),
                                  nn.Linear(64, 2 * tot))

    def _film_params(self, z):
        raw = self.film(z)                                   # (2*sum_ch,)
        gammas, betas, off = [], [], 0
        g_all, b_all = raw[: raw.numel() // 2], raw[raw.numel() // 2:]
        for c in self._film_ch:
            gammas.append(g_all[off:off + c].view(1, c, 1, 1, 1))
            betas.append(b_all[off:off + c].view(1, c, 1, 1, 1))
            off += c
        return gammas, betas

    @staticmethod
    def _film_apply(x, g, b):
        return x * (1.0 + g) + b                              # FiLM (identity at init)

    @staticmethod
    def _match(a, ref):
        if a.shape[2:] == ref.shape[2:]:
            return a
        return F.interpolate(a, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x, z):                                  # x:(1,c_in,D,H,W)  z:(zdim,)
        g, b = self._film_params(z)
        e0 = self._film_apply(self.e0(x), g[0], b[0])
        e1 = self._film_apply(self.e1(self.pool(e0)), g[1], b[1])
        m = self._film_apply(self.mid(self.pool(e1)), g[2], b[2])
        d1 = self._film_apply(self.d1(torch.cat([self._match(self.up(m), e1), e1], 1)), g[3], b[3])
        d0 = self._film_apply(self.d0(torch.cat([self._match(self.up(d1), e0), e0], 1)), g[4], b[4])
        return self.out(d0)[0, 0]                             # (D,H,W)


# ---------------------------------------------------------------- generative F-tier model
class GenerativeFTier(nn.Module):
    """Config-A F-tier with a stochastic FiLM field decoder. The encoder+scatter
    (-> input grid channels) are computed ONCE per step; only decode+physics+
    gather+eig are re-run per latent z."""

    def __init__(self, nfeat, negeo, geom, phys, mask_ch, zdim=16, width=64,
                 unet_base=16, z_mode="film", log_density=False):
        super().__init__()
        self.enc = EGNNAttnEncoder(nfeat, negeo, width=width)
        self.z_mode = z_mode
        self.zdim = zdim
        self.log_density = log_density
        base_c = width + 1 + (1 if mask_ch is not None else 0)     # latents + counts (+ mask)
        if z_mode == "film":
            self.dec = FiLMUNet3D(base_c, zdim, base=unet_base)
        else:  # concat: broadcast z as extra input channels into the validated UNet3D
            self.dec = UNet3D(c_in=base_c + zdim, base=unet_base)
        self.geom, self.phys, self.mask_ch = geom, phys, mask_ch
        self.log_amp = nn.Parameter(torch.zeros(()))
        self._idx = {"x": 0, "y": 1, "z": 2}

    def input_grid(self, h, src, dst, egeo):
        """Deterministic encoder + scatter -> stacked input channels (C,D,H,W)."""
        lat = self.enc(h, src, dst, egeo)
        grid = self.geom.scatter(lat)                             # (width,D,H,W)
        chans = [grid, self.geom_counts[None]]
        if self.mask_ch is not None:
            chans.append(self.mask_ch[None])
        return torch.cat(chans, 0)                                # (C,D,H,W)

    def decode_eig(self, x_base, z):
        """One latent z -> delta_hat -> tidal tensor -> ascending eigenvalues (N,3)."""
        if self.z_mode == "film":
            raw = self.dec(x_base[None], z)
        else:
            D, H, W = self.geom.shape
            zc = z.view(self.zdim, 1, 1, 1).expand(self.zdim, D, H, W)
            raw = self.dec(torch.cat([x_base, zc], 0)[None])
        if self.log_density:
            # decoder emits u = log(1+delta_hat) (~Gaussian; the Gaussian-latent decoder can
            # represent it). delta_hat = exp(u)-1 is skewed/lognormal and > -1 by construction —
            # fixes the SHAPE miscalibration (density is non-Gaussian). clamp(max) guards exp.
            delta = torch.exp((raw * torch.exp(self.log_amp)).clamp(max=6.0)) - 1.0
        else:
            delta = raw * torch.exp(self.log_amp)
        comps = self.phys.components(delta)
        T = torch.zeros(self.geom.n, 3, 3, device=delta.device, dtype=delta.dtype)
        for a, b in ("xx", "xy", "xz", "yy", "yz", "zz"):
            v = self.geom.gather(comps[a + b])
            T[:, self._idx[a], self._idx[b]] = v
            T[:, self._idx[b], self._idx[a]] = v
        return eigvalsh3x3(T)                                     # (N,3) ascending


def energy_score(S, y, mask):
    """Energy score over masked galaxies. S:(M,N,3) samples, y:(N,3) truth, in the
    SAME (standardized) space. Returns scalar mean ES (to minimise)."""
    Sm = S[:, mask]                                              # (M,n,3)
    ym = y[mask]                                                 # (n,3)
    M = Sm.shape[0]
    term1 = torch.linalg.norm(Sm - ym[None], dim=-1).mean(0)    # (n,)  mean_m ||s-y||
    # pairwise ||s_m - s_m'|| over ordered pairs m!=m'  (unbiased: /(M(M-1)))
    diff = Sm[:, None] - Sm[None]                                # (M,M,n,3)
    pd = torch.linalg.norm(diff, dim=-1)                        # (M,M,n)
    term2 = pd.sum((0, 1)) / (M * (M - 1))                      # (n,)
    return (term1 - 0.5 * term2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--points-xyz", type=Path, required=True)
    ap.add_argument("--gnn-arrays", type=Path, required=True)
    ap.add_argument("--scatter", choices=["cic", "tsc"], default="tsc")
    ap.add_argument("--survey-mask", action="store_true")
    ap.add_argument("--z-mode", choices=["film", "concat"], default="film")
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--m-train", type=int, default=8, help="energy-score samples per step")
    ap.add_argument("--k-eval", type=int, default=128, help="posterior samples per galaxy at eval")
    ap.add_argument("--cell-mpc", type=float, default=6.0)
    ap.add_argument("--pad-mpc", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max-train-seconds", type=float, default=12600.0,
                    help="wall-clock budget for TRAINING; break and go to eval so the "
                         "deliverable (calibration + R2) always lands within the allocation.")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-density", action="store_true",
                    help="decoder emits log(1+delta); delta=exp(u)-1 (fixes non-Gaussian density shape)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-file", type=Path, default=None)
    ap.add_argument("--samples-npz", type=Path, default=None,
                    help="optional: dump test-split posterior samples + truth for reuse")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dev = "cuda"
    if not args.smoke:
        assert torch.cuda.is_available(), (
            "gate_f3 requires CUDA but torch.cuda.is_available() is False in this "
            "process — GPU not bound to this step; abort instead of CPU-crawling.")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  z-mode={args.z_mode} zdim={args.zdim} M={args.m_train} K={args.k_eval} "
          f"scatter={args.scatter} mask={args.survey_mask}", flush=True)

    # ---- data (identical loading to gate_ftier_v2, config A) ----
    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    pos = np.load(args.points_xyz).astype(np.float64)
    arr = np.load(args.gnn_arrays)
    X = zscore(np.asarray(arr["x"], np.float64)) if "x" in arr \
        else zscore(np.asarray(cache["graph"].nodes, np.float64))
    ei = arr["edge_index"].astype(np.int64)
    eattr = np.asarray(arr["edge_attr"], np.float64) if "edge_attr" in arr else None
    print(f"nodes={len(pos)} feats={X.shape[1]} edges={ei.shape[1]} "
          f"eattr={None if eattr is None else eattr.shape}", flush=True)

    src = np.concatenate([ei[0], ei[1]]); dst = np.concatenate([ei[1], ei[0]])
    r = pos[dst] - pos[src]; d = np.linalg.norm(r, axis=1)
    los = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    rps = np.einsum("ij,ij->i", r, los[src]) / np.maximum(d, 1e-12)
    rpd = np.einsum("ij,ij->i", r, los[dst]) / np.maximum(d, 1e-12)
    egeo = [np.log(d / np.median(d)), rps, rpd, np.sqrt(np.clip(1 - rps**2, 0, 1))]
    if eattr is not None:
        ea = np.concatenate([eattr, eattr], axis=0)
        ea[:, 0] = np.log1p(ea[:, 0]); ea[:, -1] = np.log1p(np.abs(ea[:, -1]))
        egeo.append(zscore(ea))
    egeo = np.column_stack([e if e.ndim > 1 else e[:, None] for e in egeo])
    print(f"edge features negeo={egeo.shape[1]}", flush=True)

    r_gal = np.linalg.norm(pos, axis=1)
    geom = Geometry(pos, args.cell_mpc, args.pad_mpc, dev, scheme=args.scatter)
    print(f"grid {geom.shape} = {geom.numel/1e6:.1f}M cells, K={geom.K}", flush=True)
    phys = PhysicsLayer(geom.shape, args.cell_mpc, RSMOOTH_MPC, dev)
    mask_ch = None
    if args.survey_mask:
        mk = geom.survey_mask(r_gal)
        mask_ch = torch.tensor(mk, dtype=torch.float32, device=dev)
        print(f"survey mask channel: mean={float(mk.mean()):.3f}", flush=True)

    mu, sd = eig[train].mean(0), eig[train].std(0)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    h, eg = t(X), t(egeo)
    srct, dstt = t(src, torch.long), t(dst, torch.long)
    yt = t((eig - mu) / sd)                                       # standardized truth
    trm, vam = t(train, torch.bool), t(val, torch.bool)
    counts_ch = geom.counts(dev, torch.float32); counts_ch = counts_ch / counts_ch.mean().clamp(min=1e-6)
    mu_t, sd_t = t(mu), t(sd)

    model = GenerativeFTier(X.shape[1], egeo.shape[1], geom, phys, mask_ch,
                            zdim=args.zdim, width=args.width, z_mode=args.z_mode,
                            log_density=args.log_density).to(dev)
    model.geom_counts = counts_ch                                # attach for input_grid
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)

    def draw(x_base, M, gen=None):
        z = torch.randn(M, args.zdim, device=dev, generator=gen)
        return torch.stack([model.decode_eig(x_base, z[m]) for m in range(M)], 0)  # (M,N,3) raw

    if args.smoke:
        x_base = model.input_grid(h, srct, dstt, eg)
        S_raw = draw(x_base, args.m_train)                       # (M,N,3) raw eig
        S_std = (S_raw - mu_t) / sd_t
        loss = energy_score(S_std, yt, trm)
        loss.backward()
        gn = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        spread = float(S_raw.std(0).mean())                      # cross-sample spread (should be >0)
        asc = float((S_raw[..., 0] <= S_raw[..., 1]).float().mean())
        print(f"SMOKE ok: S {tuple(S_raw.shape)} ES {float(loss):.4f} grad {gn:.1f} "
              f"cross-z spread(raw) {spread:.4f} ascending-frac {asc:.3f} "
              f"finite-grad {np.isfinite(gn)}", flush=True)
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    best, best_state, pat, t0 = np.inf, None, 0, time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        x_base = model.input_grid(h, srct, dstt, eg)
        S_std = (draw(x_base, args.m_train) - mu_t) / sd_t
        loss = energy_score(S_std, yt, trm)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                xb = model.input_grid(h, srct, dstt, eg)
                Sv_raw = draw(xb, args.m_train)
                Sv = (Sv_raw - mu_t) / sd_t
                vl = float(energy_score(Sv, yt, vam))
                # cross-z spread of raw lambda1 over val galaxies (collapse monitor)
                spread = float(Sv_raw[:, vam, 0].std(0).mean())
            if vl < best - 1e-4:
                best, pat = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            if step % 250 == 0 or step == args.steps - 1:
                print(f"step {step:5d} train ES {float(loss):.4f} val ES {vl:.4f} "
                      f"lam1-spread {spread:.4f} ({time.time()-t0:.0f}s) best {best:.4f} pat {pat}",
                      flush=True)
            if pat >= args.patience:
                print(f"early stop {step}", flush=True); break
            if time.time() - t0 > args.max_train_seconds:
                print(f"wall-clock budget hit at step {step} "
                      f"({time.time()-t0:.0f}s > {args.max_train_seconds:.0f}s) -> eval now",
                      flush=True); break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- posterior: K global-grid forwards -> lambda for ALL galaxies each ----
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed + 1000)
    with torch.no_grad():
        xb = model.input_grid(h, srct, dstt, eg)
        S = torch.stack([model.decode_eig(xb, torch.randn(args.zdim, device=dev, generator=gen))
                         for _ in range(args.k_eval)], 0)        # (K,N,3) raw
    S = S.cpu().numpy().astype(np.float64)                       # (K,N,3)
    ti = np.where(test)[0]
    St = S[:, ti]                                                # (K,n_test,3) raw
    truth = eig[ti]                                             # (n_test,3) raw
    mean_f = St.mean(0)                                          # (n_test,3) posterior mean

    lines = []
    print(f"\n{'':10s} {'F3 pmean R2':>12s}  (F-tier ceiling ~0.84; G3+FMPE lam1 0.850)", flush=True)
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        r2 = r2_score(truth[:, k], mean_f[:, k])
        lines.append(f"{nm}: pmean_R2={r2:.4f}")
        print(f"{nm:10s} {r2:12.4f}", flush=True)
    clu = truth[:, 0] > 0.2
    sp = float(spearmanr(truth[clu, 0], mean_f[clu, 0]).statistic)
    lines.append(f"cluster_slice_lambda1_spearman: {sp:+.4f} (n={int(clu.sum())})")
    print(f"cluster-slice lambda1 Spearman: {sp:+.4f} (n={int(clu.sum())})", flush=True)

    # ---- calibration: SBC ranks per eigenvalue + central lambda1 coverage ----
    # SBC rank = fraction of K posterior samples below the truth (per dim). Under a
    # calibrated posterior these are Uniform(0,1); KS-test vs uniform.
    ranks = (St < truth[None]).mean(0)                          # (n_test,3) in [0,1]
    print("\ncalibration (F3 generative posterior):", flush=True)
    ks_line = []
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        p = float(kstest(ranks[:, k], "uniform").pvalue)
        ks_line.append(f"{nm}={p:.3f}")
        lines.append(f"SBC_KS_uniform_p_{nm}: {p:.4f}")
    print(f"  SBC KS-uniform p per eigenvalue: {ks_line}", flush=True)
    for q in (0.68, 0.90):
        lo_q = np.quantile(St[:, :, 0], (1 - q) / 2, axis=0)
        hi_q = np.quantile(St[:, :, 0], 1 - (1 - q) / 2, axis=0)
        cov = float(np.mean((truth[:, 0] >= lo_q) & (truth[:, 0] <= hi_q)))
        lines.append(f"lambda1_central_coverage_{int(q*100)}: {cov:.4f} (nominal {q:.2f})")
        print(f"  lambda1 central {int(q*100)}% coverage: {cov:.3f} (nominal {q:.2f})", flush=True)
    # posterior sharpness (mean central-68 width) for context
    w68 = float(np.mean(np.quantile(St[:, :, 0], 0.84, axis=0) - np.quantile(St[:, :, 0], 0.16, axis=0)))
    lines.append(f"lambda1_mean_central68_width: {w68:.4f}")
    print(f"  lambda1 mean 68% width: {w68:.4f}  ascending-frac(all samples): "
          f"{float((St[..., 0] <= St[..., 1]).mean()):.3f}", flush=True)

    if args.samples_npz is not None:
        args.samples_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.samples_npz, samples_test=St.astype(np.float32),
                            truth_test=truth.astype(np.float64), test_index=ti,
                            mu=mu, sd=sd, seed=args.seed)
        print(f"posterior samples saved: {St.shape} -> {args.samples_npz}", flush=True)
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(
            f"gate_f3_generative_ftier z-mode={args.z_mode} zdim={args.zdim} M={args.m_train} "
            f"K={args.k_eval} scatter={args.scatter} mask={args.survey_mask} seed={args.seed}\n"
            + "\n".join(lines) + "\n")
        json.dump({"lines": lines, "args": {k: str(v) for k, v in vars(args).items()}},
                  open(args.out_file.with_suffix(".json"), "w"), indent=2)
        print(f"summary written: {args.out_file}", flush=True)

    print("\nGATE F3: generative F-tier is a CALIBRATED posterior IF SBC KS-p not tiny "
          "AND lambda1 coverage near nominal. Compare pmean R2 to the ~0.84 F-tier "
          "ceiling and to G3+FMPE (P3 lambda1 0.850, cov 0.594@68 pre-tempering).", flush=True)


if __name__ == "__main__":
    main()
