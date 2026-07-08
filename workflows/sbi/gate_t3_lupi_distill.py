#!/usr/bin/env python3
"""Gate T3 — LUPI / privileged-information distillation (field-level plan, row T3).

Honest "multimodal training, unimodal inference": a TEACHER that SEES the sim
density field (privileged information available only at TRAIN time) regularizes
the STUDENT graph model's latents; the teacher is ABSENT at inference. This is
Learning Using Privileged Information (Vapnik) / generalized distillation
(Lopez-Paz et al. 2016). Expected gain is MODEST — our sims are not data-poor
and the density field deterministically implies the tidal target via Poisson, so
the teacher is near-perfect. The point is a disciplined measurement, not a win.

DESIGN
------
* Student  : the EGNN-lite graph model from `gate_g4_egnn_smoke.py`
  (reimplemented here, faithful, so this script is self-contained and can return
  a node latent). At INFERENCE it uses ONLY galaxy features + observer-frame
  geometry + edges. Predicts the 3 standardized eigenvalues; identical
  masks/splits/standardization/eval as the G4 smoke gate.
* Teacher  : a small 3-D CNN over an (PxPxP) patch of the BOX-FRAME sim density
  field centred on each galaxy's TRUE host-halo voxel -> latent -> 3 eigenvalues.
  Trained on the train split ONLY, then FROZEN; its node latents + softened
  eigenvalue predictions are cached (detached) and used to regularize the
  student. Never present at eval.
* Distill  : student total = MSE(student_eig, target) + alpha * L_distill on
  TRAIN nodes only. L_distill in {cosine, l2} pulls a projected student latent
  toward the teacher latent; {soft-eig} matches the teacher's eigenvalue
  outputs (generalized distillation); {both} = cosine + soft-eig.

DISCIPLINE (the deliverable)
----------------------------
>=3 seeds for BOTH the distilled student and a matched no-distillation control
(alpha=0), identical everything else (same student init per seed). The result is
Delta lambda1 R^2 (distilled - control) vs seed scatter. A gain smaller than seed
noise = NULL = close idea 1.

BOX-FRAME MAPPING (critical; see plan + memory gotcha)
------------------------------------------------------
The galaxy points_xyz are OBSERVER-frame comoving (|xyz|~1e3 Mpc) and DO NOT
index the sim grid. Each galaxy's true box-frame voxel is recovered from its host
halo: (FILE_NUM, HALO_INDEX) -> CompaSOHaloCatalog(halo_info_XXX.asdf,
fields=['x_com'], convert_units=True, cleaned=False) -> x_com (Mpc/h box frame)
-> floor(mod(x_com, BOX)/cell), ngrid=2048, BOX=2000 Mpc/h. Reuses the exact
logic of `annotate_cutsky_with_tweb_eigs.py` pass-2 and
`export_cutsky_boxframe_points.py`. A density-sanity check (galaxy voxels must sit
at ~3x random median density, >90% above random median) runs before any
training; if it FAILS the box mapping is wrong and the script STOPS.

Torch; GPU if available. Nothing writes to any existing/shared file — the only
repo output is this script; all artifacts go under the T3 scratch dir.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# ---- default paths (all read-only inputs; outputs under T3_DIR) -------------
FITS = ("/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
        "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_"
        "wedge_targets.fits")
CACHE = ("/pscratch/sd/d/dkololgi/abacus/sbi_caches/path1_fiberassign_mock_bgs_maglim_"
         "rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_sbi_cache_3d_lineareig_si.pkl")
POINTS_XYZ = ("/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
              "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_"
              "points_xyz.npy")
GNN_ARRAYS = ("/pscratch/sd/d/dkololgi/abacus/graph_constructions/wedges/path1_fiberassign/"
              "path1_fiberassign_mock_bgs_maglim_rs7_wedge_ra120_160_dec14p5_30p6_z0p2_0p3_"
              "cugraph_gnn_arrays.npz")
DENSITY = ("/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/"
           "AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy")
HALO_INFO_DIR = ("/pscratch/sd/d/dkololgi/AbacusSummit_densities/AbacusSummit_base_c000_ph000/"
                 "halos/z0.200/halo_info")
T3_DIR = Path("/pscratch/sd/d/dkololgi/abacus/field_level_tests/T3")

NGRID = 2048
BOXSIZE = 2000.0  # Mpc/h


# ============================ box-frame voxels ===============================
def to_grid_indices(xyz: np.ndarray, ngrid: int = NGRID, boxsize: float = BOXSIZE):
    """Replicates annotate_cutsky_with_tweb_eigs.to_grid_indices (box frame)."""
    cell = boxsize / ngrid
    m = np.mod(xyz, boxsize)
    ix = np.clip(np.floor(m[:, 0] / cell).astype(np.int64), 0, ngrid - 1)
    iy = np.clip(np.floor(m[:, 1] / cell).astype(np.int64), 0, ngrid - 1)
    iz = np.clip(np.floor(m[:, 2] / cell).astype(np.int64), 0, ngrid - 1)
    return np.column_stack([ix, iy, iz])


def compute_box_voxels(fits_path, halo_info_dir, halo_pos_field="x_com"):
    """(FILE_NUM, HALO_INDEX) -> host-halo x_com -> box voxel indices (N,3).

    Uses cleaned=False and x_com to match the indexing under which HALO_INDEX
    was written by annotate_cutsky_with_tweb_eigs pass-2.
    """
    import fitsio
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    t = fitsio.read(fits_path)
    file_num = np.asarray(t["FILE_NUM"], dtype=np.int64)
    halo_index = np.asarray(t["HALO_INDEX"], dtype=np.int64)
    n = len(file_num)
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    hdir = Path(halo_info_dir)
    for fn in np.unique(file_num):
        sel = file_num == fn
        hp = hdir / f"halo_info_{int(fn):03d}.asdf"
        if not hp.exists():
            raise FileNotFoundError(f"Missing halo_info file: {hp}")
        cat = CompaSOHaloCatalog(str(hp), fields=[halo_pos_field], subsamples=False,
                                 convert_units=True, verbose=False, cleaned=False)
        pos = np.asarray(cat.halos[halo_pos_field], dtype=np.float64)  # Mpc/h box frame
        nh = pos.shape[0]
        hidx = halo_index[sel]
        ok = (hidx >= 0) & (hidx < nh)
        rows = np.where(sel)[0]
        xyz[rows[ok]] = pos[hidx[ok]]
        if np.any(~ok):
            print(f"  WARNING file_num={int(fn)}: {int((~ok).sum())} invalid HALO_INDEX "
                  f"(nhalo={nh})")
        del cat, pos
    n_bad = int(np.sum(~np.isfinite(xyz).all(axis=1)))
    if n_bad:
        print(f"  WARNING: {n_bad}/{n} rows lack a valid halo xyz (kept as clip-to-0).")
    vox = to_grid_indices(np.nan_to_num(xyz, nan=0.0))
    return vox.astype(np.int64), n_bad


# =========================== density-field patches ===========================
def density_sanity(dens_mm, vox, rng, n_rand=100_000):
    """Galaxy voxels must sit at ELEVATED density vs random field voxels."""
    g = dens_mm[vox[:, 0], vox[:, 1], vox[:, 2]].astype(np.float64)
    ri = rng.integers(0, NGRID, size=(n_rand, 3))
    r = dens_mm[ri[:, 0], ri[:, 1], ri[:, 2]].astype(np.float64)
    g_med, r_med = float(np.median(g)), float(np.median(r))
    frac_above = float(np.mean(g > r_med))
    ratio = g_med / r_med if r_med > 0 else float("inf")
    res = dict(galaxy_median_density=g_med, random_median_density=r_med,
               galaxy_mean_density=float(g.mean()), random_mean_density=float(r.mean()),
               ratio_galaxy_over_random_median=ratio, frac_galaxy_above_random_median=frac_above,
               n_galaxies=int(len(vox)), n_random=int(n_rand))
    return res


def extract_patches(dens_mm, vox, patch, chunk=4000):
    """(N,1,P,P,P) periodic density patches centred on each galaxy voxel."""
    n = len(vox)
    off = np.arange(-(patch // 2), patch - patch // 2, dtype=np.int64)  # e.g. -4..3 for P=8
    out = np.empty((n, patch, patch, patch), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ix = (vox[s:e, 0, None] + off[None, :]) % NGRID  # (nc,P)
        iy = (vox[s:e, 1, None] + off[None, :]) % NGRID
        iz = (vox[s:e, 2, None] + off[None, :]) % NGRID
        out[s:e] = dens_mm[ix[:, :, None, None], iy[:, None, :, None], iz[:, None, None, :]]
        if s == 0 or e == n:
            print(f"  patches {e:,}/{n:,}")
    return out[:, None, :, :, :]  # add channel dim


# ================================ teacher ====================================
class TeacherCNN(nn.Module):
    """Small 3-D CNN over a density patch -> latent -> 3 eigenvalues."""

    def __init__(self, patch, latent_dim=32, width=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, 2 * width, 3, padding=1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.to_latent = nn.Sequential(nn.Linear(2 * width, latent_dim), nn.SiLU())
        self.head = nn.Linear(latent_dim, 3)

    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        z = self.to_latent(h)
        return self.head(z), z


# ================================ student ====================================
class StudentEGNN(nn.Module):
    """EGNN-lite (faithful copy of gate_g4_egnn_smoke.EGNNLite) that also returns
    the pre-head node latent for distillation."""

    def __init__(self, nfeat, negeo, width=96, layers=5, aggregation="mean", heads=4):
        super().__init__()
        assert width % heads == 0
        self.aggregation, self.heads, self.width = aggregation, heads, width
        self.embed = nn.Linear(nfeat, width)
        self.msg = nn.ModuleList()
        self.upd = nn.ModuleList()
        self.att = nn.ModuleList()
        for _ in range(layers):
            self.msg.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, width), nn.SiLU()))
            self.upd.append(nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                          nn.Linear(width, width)))
            self.att.append(nn.Sequential(nn.Linear(2 * width + negeo, width), nn.SiLU(),
                                          nn.Linear(width, heads)))
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def _segment_softmax(self, logits, dst, n):
        mx = torch.full((n, logits.shape[1]), -1e30, device=logits.device, dtype=logits.dtype)
        mx.scatter_reduce_(0, dst[:, None].expand(-1, logits.shape[1]), logits,
                           reduce="amax", include_self=True)
        w = torch.exp(logits - mx[dst])
        den = torch.zeros_like(mx).index_add_(0, dst, w)
        return w / den[dst].clamp(min=1e-12)

    def _layer(self, h, src, dst, egeo, msg, upd, att):
        n = h.shape[0]
        pair = torch.cat([h[src], h[dst], egeo], dim=1)
        m = msg(pair)
        if self.aggregation == "attention":
            alpha = self._segment_softmax(att(pair), dst, n)
            E, W = m.shape
            mh = m.view(E, self.heads, W // self.heads) * alpha[:, :, None]
            agg = torch.zeros(n, self.heads, W // self.heads, device=h.device, dtype=m.dtype)
            agg.index_add_(0, dst, mh)
            agg = agg.view(n, W)
        else:
            agg = torch.zeros(n, m.shape[1], device=h.device, dtype=m.dtype)
            cnt = torch.zeros(n, 1, device=h.device, dtype=m.dtype)
            agg.index_add_(0, dst, m)
            cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device, dtype=m.dtype))
            agg = agg / cnt.clamp(min=1)
        return h + upd(torch.cat([h, agg], dim=1))

    def forward(self, h, src, dst, egeo):
        h = self.embed(h)
        for msg, upd, att in zip(self.msg, self.upd, self.att):
            h = checkpoint(self._layer, h, src, dst, egeo, msg, upd, att, use_reentrant=False)
        return self.head(h), h  # (eig prediction, node latent)


# ============================== data loading =================================
def load_data(args, dev):
    cache = pickle.load(open(args.cache, "rb"))
    eig = np.asarray(cache["eigenvalues_raw"], np.float64)
    train, val, test = (np.asarray(m).astype(bool) for m in cache["masks"])
    X = np.asarray(cache["graph"].nodes, np.float64)

    pos = np.load(args.points_xyz).astype(np.float64)
    ei = np.load(args.gnn_arrays)["edge_index"].astype(np.int64)
    src = np.concatenate([ei[0], ei[1]]); dst = np.concatenate([ei[1], ei[0]])
    r = pos[dst] - pos[src]
    d = np.linalg.norm(r, axis=1)
    los = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    rpar_s = np.einsum("ij,ij->i", r, los[src]) / np.maximum(d, 1e-12)
    rpar_d = np.einsum("ij,ij->i", r, los[dst]) / np.maximum(d, 1e-12)
    egeo = np.column_stack([np.log(d / np.median(d)), rpar_s, rpar_d,
                            np.sqrt(np.clip(1 - rpar_s ** 2, 0, 1))])

    mu, sd = eig[train].mean(0), eig[train].std(0)
    Y = (eig - mu) / sd
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    data = dict(
        eig=eig, mu=mu, sd=sd,
        h=t(X), yt=t(Y),
        src=t(src, torch.long), dst=t(dst, torch.long), eg=t(egeo),
        trm=t(train, torch.bool), vam=t(val, torch.bool), tem=t(test, torch.bool),
        train=train, val=val, test=test, nfeat=X.shape[1], negeo=egeo.shape[1],
    )
    print(f"nodes={len(X)}, directed edges={len(src)}, nfeat={X.shape[1]}, "
          f"egeo={egeo.shape[1]}; train/val/test={train.sum()}/{val.sum()}/{test.sum()}")
    return data


# ============================ teacher pretrain ===============================
def train_teacher(patches_t, data, args, dev):
    """Pretrain teacher CNN on train nodes; return detached (latent, eig) for all
    nodes plus its own test R^2 (a diagnostic of how privileged it is)."""
    teacher = TeacherCNN(args.patch, latent_dim=args.latent_dim).to(dev)
    opt = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr)
    yt, trm, vam = data["yt"], data["trm"], data["vam"]
    tr_idx = torch.where(trm)[0]
    best_val, best_state, patience = np.inf, None, 0
    n_tr = len(tr_idx)
    for step in range(args.teacher_steps):
        teacher.train()
        perm = tr_idx[torch.randperm(n_tr, device=dev)[:args.teacher_batch]]
        opt.zero_grad()
        pe, _ = teacher(patches_t[perm])
        loss = ((pe - yt[perm]) ** 2).mean()
        loss.backward(); opt.step()
        if step % 25 == 0 or step == args.teacher_steps - 1:
            teacher.eval()
            with torch.no_grad():
                vi = torch.where(vam)[0]
                vloss = 0.0
                for s in range(0, len(vi), 16384):
                    idx = vi[s:s + 16384]
                    pv, _ = teacher(patches_t[idx])
                    vloss += float(((pv - yt[idx]) ** 2).sum())
                vloss /= len(vi)
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
            else:
                patience += 1
            if step % 100 == 0:
                print(f"    teacher step {step:4d} train {float(loss):.4f} val {vloss:.4f}")
            if patience >= 10:
                print(f"    teacher early stop at step {step}")
                break
    teacher.load_state_dict(best_state)
    teacher.eval()
    n = patches_t.shape[0]
    lat = torch.zeros(n, args.latent_dim, device=dev)
    peig = torch.zeros(n, 3, device=dev)
    with torch.no_grad():
        for s in range(0, n, 16384):
            e = min(s + 16384, n)
            pe, z = teacher(patches_t[s:e])
            peig[s:e], lat[s:e] = pe, z
    # teacher test R^2 (privilege diagnostic)
    tem = data["tem"].cpu().numpy()
    pe_phys = peig.cpu().numpy() * data["sd"] + data["mu"]
    tr2 = [float(r2_score(data["eig"][tem, k], pe_phys[tem, k])) for k in range(3)]
    print(f"    teacher test R^2 (privileged): "
          f"l1={tr2[0]:.3f} l2={tr2[1]:.3f} l3={tr2[2]:.3f}")
    return lat.detach(), peig.detach(), tr2


# ============================ student training ===============================
def distill_loss(proj, s_lat, s_eig, teach_lat, teach_eig, trm, target):
    """L_distill on TRAIN nodes only."""
    sl = proj(s_lat[trm])
    tl = teach_lat[trm]
    terms = []
    if target in ("cosine", "both"):
        terms.append((1.0 - torch.cosine_similarity(sl, tl, dim=1)).mean())
    if target == "l2":
        sln = sl / sl.norm(dim=1, keepdim=True).clamp(min=1e-8)
        tln = tl / tl.norm(dim=1, keepdim=True).clamp(min=1e-8)
        terms.append(((sln - tln) ** 2).sum(1).mean())
    if target in ("soft-eig", "both"):
        terms.append(((s_eig[trm] - teach_eig[trm]) ** 2).mean())
    return sum(terms)


def train_student(data, args, dev, seed, alpha, teach_lat=None, teach_eig=None):
    torch.manual_seed(seed)
    model = StudentEGNN(data["nfeat"], data["negeo"], aggregation=args.aggregation,
                        heads=args.heads).to(dev)
    proj = nn.Linear(model.width, args.latent_dim).to(dev)
    params = list(model.parameters()) + (list(proj.parameters()) if alpha > 0 else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    h, yt, src, dst, eg = data["h"], data["yt"], data["src"], data["dst"], data["eg"]
    trm, vam = data["trm"], data["vam"]
    best_val, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for step in range(args.steps):
        model.train(); opt.zero_grad()
        out, lat = model(h, src, dst, eg)
        eig_loss = ((out[trm] - yt[trm]) ** 2).mean()
        loss = eig_loss
        if alpha > 0:
            loss = loss + alpha * distill_loss(proj, lat, out, teach_lat, teach_eig,
                                               trm, args.distill_target)
        loss.backward(); opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vout, _ = model(h, src, dst, eg)
                vloss = float(((vout[vam] - yt[vam]) ** 2).mean())  # eig-only val
            if vloss < best_val - 1e-4:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if step % 500 == 0:
                print(f"    student(a={alpha}) step {step:5d} eig {float(eig_loss):.4f} "
                      f"val {vloss:.4f} ({time.time()-t0:.0f}s)")
            if patience >= 12:
                print(f"    student(a={alpha}) early stop at step {step}")
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = (model(h, src, dst, eg)[0].cpu().numpy()) * data["sd"] + data["mu"]
    tem = data["test"]
    ti = np.where(tem)[0]
    r2 = [float(r2_score(data["eig"][ti, k], pred[ti, k])) for k in range(3)]
    clu = data["eig"][ti, 0] > 0.2
    sp = float(spearmanr(data["eig"][ti, 0][clu], pred[ti, 0][clu]).statistic)
    return dict(r2=r2, cluster_lambda1_spearman=sp, n_cluster=int(clu.sum()))


# ================================== main =====================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=FITS)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--points-xyz", default=POINTS_XYZ)
    ap.add_argument("--gnn-arrays", default=GNN_ARRAYS)
    ap.add_argument("--density", default=DENSITY)
    ap.add_argument("--halo-info-dir", default=HALO_INFO_DIR)
    ap.add_argument("--out-dir", type=Path, default=T3_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="distillation weight for the distilled student (control uses 0).")
    ap.add_argument("--distill-target", choices=["cosine", "l2", "soft-eig", "both"],
                    default="cosine")
    ap.add_argument("--patch", type=int, default=8, help="density patch side (voxels).")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--teacher-steps", type=int, default=2000)
    ap.add_argument("--teacher-lr", type=float, default=1e-3)
    ap.add_argument("--teacher-batch", type=int, default=8192)
    ap.add_argument("--aggregation", choices=["mean", "attention"], default="mean")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--grid-mean", type=float, default=3.844,
                    help="global mean of the density grid; delta = dens/mean - 1.")
    ap.add_argument("--sanity-only", action="store_true",
                    help="compute box voxels + density sanity, then STOP (no training).")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- box voxels (cached) ----
    vox_path = args.out_dir / "box_voxels.npy"
    if vox_path.exists():
        vox = np.load(vox_path)
        print(f"loaded cached box voxels {vox.shape} from {vox_path}")
        n_bad = int(np.load(args.out_dir / "box_voxels_nbad.npy")) if (
            args.out_dir / "box_voxels_nbad.npy").exists() else 0
    else:
        print("computing box-frame voxels from halo x_com ...")
        vox, n_bad = compute_box_voxels(args.fits, args.halo_info_dir)
        np.save(vox_path, vox)
        np.save(args.out_dir / "box_voxels_nbad.npy", np.array(n_bad))
        print(f"saved {vox_path} ({vox.shape}); n_bad={n_bad}")

    # ---- density sanity (mandatory gate) ----
    print("opening density grid (memmap) ...")
    dens_mm = np.load(args.density, mmap_mode="r")
    rng = np.random.default_rng(0)
    sanity = density_sanity(dens_mm, vox, rng)
    sanity["n_bad_halo_link"] = n_bad
    print("DENSITY SANITY:")
    for k, v in sanity.items():
        print(f"  {k}: {v}")
    sanity_ok = (sanity["ratio_galaxy_over_random_median"] >= 2.0 and
                 sanity["frac_galaxy_above_random_median"] >= 0.85)
    sanity["passed"] = bool(sanity_ok)
    (args.out_dir / "density_sanity.json").write_text(json.dumps(sanity, indent=2))
    if not sanity_ok:
        print("\n*** DENSITY SANITY FAILED — box mapping is likely wrong. STOPPING. ***")
        print("    (expected galaxy median ~3x random median, >90% above). "
              "Do NOT train on a broken teacher.")
        return
    print("density sanity PASSED.\n")
    if args.sanity_only:
        print("--sanity-only set; stopping before training.")
        return

    # ---- patches (cached) ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    patch_path = args.out_dir / f"patches_P{args.patch}.npy"
    if patch_path.exists():
        patches = np.load(patch_path)
        print(f"loaded cached patches {patches.shape}")
    else:
        print(f"extracting {args.patch}^3 density patches ...")
        patches = extract_patches(dens_mm, vox, args.patch)
        np.save(patch_path, patches)
        print(f"saved {patch_path} ({patches.shape})")
    # standardize patches as delta, then z-score over train split
    data = load_data(args, dev)
    delta = patches.astype(np.float32) / args.grid_mean - 1.0
    tr = data["train"]
    pm, ps = float(delta[tr].mean()), float(delta[tr].std() + 1e-8)
    delta = (delta - pm) / ps
    patches_t = torch.tensor(delta, dtype=torch.float32, device=dev)
    del patches, delta

    # ---- multi-seed control vs distilled ----
    results = {"seeds": {}, "config": vars(args).copy()}
    results["config"] = {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in results["config"].items()}
    results["density_sanity"] = sanity
    for seed in args.seeds:
        print(f"\n===== SEED {seed} =====")
        print("  pretraining teacher ...")
        torch.manual_seed(seed)
        teach_lat, teach_eig, teacher_r2 = train_teacher(patches_t, data, args, dev)
        print("  control student (alpha=0) ...")
        ctrl = train_student(data, args, dev, seed, 0.0)
        print(f"    control  R^2 l1/l2/l3 = "
              f"{ctrl['r2'][0]:.3f}/{ctrl['r2'][1]:.3f}/{ctrl['r2'][2]:.3f}")
        print(f"  distilled student (alpha={args.alpha}, target={args.distill_target}) ...")
        dist = train_student(data, args, dev, seed, args.alpha, teach_lat, teach_eig)
        print(f"    distilled R^2 l1/l2/l3 = "
              f"{dist['r2'][0]:.3f}/{dist['r2'][1]:.3f}/{dist['r2'][2]:.3f}")
        results["seeds"][str(seed)] = dict(teacher_test_r2=teacher_r2,
                                           control=ctrl, distilled=dist)

    # ---- aggregate ----
    def stk(cond, k):
        return np.array([results["seeds"][str(s)][cond]["r2"][k] for s in args.seeds])
    agg = {}
    print("\n================ AGGREGATE (mean +/- std over seeds) ================")
    print(f"{'':10s} {'control':>18s} {'distilled':>18s} {'Delta':>14s}")
    for k, nm in enumerate(["lambda1", "lambda2", "lambda3"]):
        c, d = stk("control", k), stk("distilled", k)
        delta_seed = d - c
        agg[nm] = dict(control_mean=float(c.mean()), control_std=float(c.std()),
                       distilled_mean=float(d.mean()), distilled_std=float(d.std()),
                       delta_mean=float(delta_seed.mean()), delta_std=float(delta_seed.std()),
                       control_per_seed=c.tolist(), distilled_per_seed=d.tolist())
        print(f"{nm:10s} {c.mean():7.3f}+/-{c.std():.3f}   "
              f"{d.mean():7.3f}+/-{d.std():.3f}   {delta_seed.mean():+7.3f}+/-{delta_seed.std():.3f}")
    results["aggregate"] = agg

    # GATE read on lambda1
    d1 = agg["lambda1"]
    # seed scatter = std of the paired delta across seeds (its own null band)
    gate_pass = abs(d1["delta_mean"]) > d1["delta_std"] and d1["delta_mean"] > 0
    results["gate"] = dict(
        delta_lambda1_mean=d1["delta_mean"], delta_lambda1_seed_scatter=d1["delta_std"],
        control_lambda1_seed_std=d1["control_std"],
        verdict=("KEEP LUPI (delta>scatter)" if gate_pass else
                 "NULL — close idea 1 (delta within seed scatter)"))
    print(f"\nGATE: Delta lambda1 R^2 = {d1['delta_mean']:+.3f} "
          f"(seed scatter +/-{d1['delta_std']:.3f}); "
          f"control seed std +/-{d1['control_std']:.3f}")
    print(f"VERDICT: {results['gate']['verdict']}")

    out_json = args.out_json or (args.out_dir / "t3_lupi_results.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nresults written: {out_json}")


if __name__ == "__main__":
    main()
