import argparse
import sys
from pathlib import Path

import numpy as np
from shift import mpiutils

# Headless plotting for batch/interactive compute-node runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_paths import ABACUS_TWEB_OUTPUT_DIR
from shared.resource_requirements import require_cpu_mpi_slurm
from workflows.abacus_tweb.abacus_process_particles2 import run_tweb_memory_optimized


DEFAULT_DENSITY_PATH = (
    "/pscratch/sd/d/dkololgi/AbacusSummit_densities/density_fields/"
    "AbacusSummit_base_c000_ph000_z0.200_ngrid_2048_10pc_density_field.npy"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MPI T-Web directly from a full density field .npy (no pre-slab files). "
            "Supports cactus native mpi_run_tweb and memory-optimized backend."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "plot"],
        default="generate",
        help="generate: run MPI T-web. plot: only plot an existing run directory.",
    )
    parser.add_argument(
        "--density-path",
        type=str,
        default=DEFAULT_DENSITY_PATH,
        help="Path to full 3D density field .npy file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{ABACUS_TWEB_OUTPUT_DIR}/fullgrid_runs",
        help="Root output directory for rank outputs and plots.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["optimized", "native"],
        default="optimized",
        help="T-Web backend: optimized (custom) or native (cactus mpi_run_tweb).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Eigenvalue threshold for web classification.",
    )
    parser.add_argument(
        "--rsmooth",
        type=float,
        nargs="+",
        default=[2.0],
        help="One or more Gaussian smoothing scales [Mpc/h]. Example: --rsmooth 2 4 8",
    )
    parser.add_argument(
        "--boxsize",
        type=float,
        default=2000.0,
        help="Simulation box size [Mpc/h].",
    )
    parser.add_argument(
        "--plot-slices",
        action="store_true",
        help="Generate rank-0 local diagnostic slice plots during generate mode.",
    )
    parser.add_argument(
        "--stitch-cube",
        action="store_true",
        help="In generate mode, stitch rank outputs into full cweb cube on rank 0.",
    )
    parser.add_argument(
        "--plot-run-dir",
        type=str,
        default="",
        help="In plot mode, existing run directory containing rank outputs and/or stitched cube.",
    )
    parser.add_argument(
        "--slice-fracs",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
        help="Global x-slice fractions for plotting in plot mode.",
    )
    return parser.parse_args()


def _x_bounds_for_rank(rank, size, ngrid):
    nx_base = ngrid // size
    remainder = ngrid % size
    if rank < remainder:
        x_start = rank * (nx_base + 1)
        x_end = x_start + (nx_base + 1)
    else:
        x_start = rank * nx_base + remainder
        x_end = x_start + nx_base
    return x_start, x_end


def _load_local_density_from_full(density_path, boxsize, MPI):
    rank, size = MPI.rank, MPI.size

    dens = np.load(density_path, mmap_mode="r")
    if dens.ndim != 3 or dens.shape[0] != dens.shape[1] or dens.shape[1] != dens.shape[2]:
        raise ValueError(f"Expected cubic 3D density array, got shape={dens.shape}")
    ngrid = int(dens.shape[0])

    x_start, x_end = _x_bounds_for_rank(rank, size, ngrid)
    dens_local = np.ascontiguousarray(dens[x_start:x_end, :, :], dtype=np.float32)
    del dens

    if rank == 0:
        print(f"Density path: {density_path}")
        print(f"Full grid: {ngrid}^3, boxsize={boxsize}")
        print(f"MPI ranks: {size}")
    print(f"[rank {rank}] Local slab x=[{x_start}, {x_end}), shape={dens_local.shape}")
    return dens_local, ngrid, x_start, x_end


def _run_native_mpi_tweb(dens_local, boxsize, ngrid, threshold, MPI, rsmooth, verbose):
    import cactus

    # Try verbose signature first, then fallback for API variants.
    try:
        return cactus.src.tweb.mpi_run_tweb(
            dens_local, boxsize, ngrid, threshold, MPI, Rsmooth=rsmooth, verbose=verbose
        )
    except TypeError:
        return cactus.src.tweb.mpi_run_tweb(
            dens_local, boxsize, ngrid, threshold, MPI, Rsmooth=rsmooth
        )


def _run_backend(backend, dens_local, boxsize, ngrid, threshold, MPI, rsmooth, verbose):
    if backend == "optimized":
        return run_tweb_memory_optimized(
            dens_local, boxsize, ngrid, threshold, MPI, Rsmooth=rsmooth, verbose=verbose
        )
    return _run_native_mpi_tweb(
        dens_local, boxsize, ngrid, threshold, MPI, rsmooth=rsmooth, verbose=verbose
    )


def _run_dir(root, backend, ngrid, rsmooth):
    rsmooth_tag = "none" if rsmooth is None else f"{rsmooth:g}"
    return Path(root) / f"backend_{backend}_ngrid_{ngrid}_rsmooth_{rsmooth_tag}"


def _rank_output_path(run_dir, rank):
    return run_dir / f"abacus_cactus_tweb_rank{rank:04d}.npz"


def _index_rank_outputs(run_dir):
    files = sorted(run_dir.glob("abacus_cactus_tweb_rank*.npz"))
    if not files:
        return []

    out = []
    for fp in files:
        with np.load(fp) as d:
            if "x_start" not in d or "x_end" not in d:
                raise KeyError(f"{fp} missing x_start/x_end; cannot stitch or global-slice.")
            out.append(
                {
                    "path": fp,
                    "x_start": int(d["x_start"]),
                    "x_end": int(d["x_end"]),
                    "ngrid": int(d["ngrid"]),
                    "boxsize": float(d["boxsize"]),
                    "threshold": float(d["threshold"]),
                    "Rsmooth": float(d["Rsmooth"]),
                }
            )
    out.sort(key=lambda x: x["x_start"])
    return out


def _load_global_cweb_slice(indexed, global_ix):
    for meta in indexed:
        if meta["x_start"] <= global_ix < meta["x_end"]:
            local_ix = global_ix - meta["x_start"]
            with np.load(meta["path"]) as d:
                return np.asarray(d["cweb"][local_ix, :, :], dtype=np.uint8)
    raise ValueError(f"Global ix={global_ix} not covered by rank outputs.")


def _stitch_cweb_cube(run_dir, indexed):
    if not indexed:
        print("[rank 0] No rank outputs found to stitch.")
        return None

    ngrid = indexed[0]["ngrid"]
    out_path = run_dir / "cweb_full.npy"
    cweb_full = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(ngrid, ngrid, ngrid)
    )
    for meta in indexed:
        with np.load(meta["path"]) as d:
            cweb_local = np.asarray(d["cweb"], dtype=np.uint8)
        cweb_full[meta["x_start"]:meta["x_end"], :, :] = cweb_local
    cweb_full.flush()
    del cweb_full
    print(f"[rank 0] Stitched full cube: {out_path}")
    return out_path


def _plot_local_diagnostics(run_dir, cweb_local, eig_vals_local, ngrid, boxsize, threshold, rsmooth):
    nx_local = cweb_local.shape[0]
    if nx_local < 1:
        print("Empty local chunk; skipping plots.")
        return

    ix_vals = sorted(
        {int(np.clip(round(f * (nx_local - 1)), 0, nx_local - 1)) for f in [0.25, 0.5, 0.75]}
    )

    fig, axes = plt.subplots(1, len(ix_vals), figsize=(5.0 * len(ix_vals), 4.4), constrained_layout=True)
    if len(ix_vals) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis", 4)
    for ax, ixl in zip(axes, ix_vals):
        cweb_slice = np.asarray(cweb_local[ixl, :, :], dtype=np.float32)
        im = ax.imshow(cweb_slice.T, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"CWEB local slice (ix={ixl})")
        ax.set_xlabel("grid y")
        ax.set_ylabel("grid z")
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cb.set_label("Environment class")
    fig.suptitle(
        f"CWEB Local Slices (ngrid={ngrid}, box={boxsize:.0f}, threshold={threshold}, Rsmooth={rsmooth:g})",
        fontsize=12,
    )
    cweb_png = run_dir / "cweb_slices_rank0_local.png"
    fig.savefig(cweb_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    ix_mid = nx_local // 2
    eig_slices = [eig_vals_local[0, ix_mid], eig_vals_local[1, ix_mid], eig_vals_local[2, ix_mid]]
    vabs = max(float(np.nanmax(np.abs(s))) for s in eig_slices)
    vabs = max(vabs, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for i, ax in enumerate(axes):
        im = ax.imshow(
            eig_slices[i].T,
            origin="lower",
            cmap="coolwarm",
            vmin=-vabs,
            vmax=vabs,
            interpolation="nearest",
        )
        ax.set_title(f"lambda{i+1} local (ix={ix_mid})")
        ax.set_xlabel("grid y")
        ax.set_ylabel("grid z")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Eigenvalue")
    fig.suptitle(
        f"T-Web Eigenvalue Local Slices (ngrid={ngrid}, threshold={threshold}, Rsmooth={rsmooth:g})",
        fontsize=12,
    )
    eig_png = run_dir / "eig_slices_rank0_local.png"
    fig.savefig(eig_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[rank 0] Saved plots:\n  - {cweb_png}\n  - {eig_png}")


def _plot_existing_run(run_dir, slice_fracs):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cweb_full_path = run_dir / "cweb_full.npy"
    if cweb_full_path.exists():
        cweb = np.load(cweb_full_path, mmap_mode="r")
        ngrid = cweb.shape[0]
        ix_vals = sorted({int(np.clip(round(f * (ngrid - 1)), 0, ngrid - 1)) for f in slice_fracs})

        fig, axes = plt.subplots(1, len(ix_vals), figsize=(5.0 * len(ix_vals), 4.4), constrained_layout=True)
        if len(ix_vals) == 1:
            axes = [axes]
        cmap = plt.get_cmap("viridis", 4)
        for ax, ixg in zip(axes, ix_vals):
            im = ax.imshow(cweb[ixg].T, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax.set_title(f"CWEB global slice (ix={ixg})")
            ax.set_xlabel("grid y")
            ax.set_ylabel("grid z")
            cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
            cb.set_label("Environment class")
        fig.suptitle(f"CWEB Global Slices (ngrid={ngrid})", fontsize=12)
        out_png = run_dir / "cweb_slices_global.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")
        return

    indexed = _index_rank_outputs(run_dir)
    if not indexed:
        raise FileNotFoundError(
            f"No stitched cube or rank outputs found in {run_dir}."
        )
    ngrid = indexed[0]["ngrid"]
    threshold = indexed[0]["threshold"]
    rsmooth = indexed[0]["Rsmooth"]

    ix_vals = sorted({int(np.clip(round(f * (ngrid - 1)), 0, ngrid - 1)) for f in slice_fracs})
    fig, axes = plt.subplots(1, len(ix_vals), figsize=(5.0 * len(ix_vals), 4.4), constrained_layout=True)
    if len(ix_vals) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis", 4)
    for ax, ixg in zip(axes, ix_vals):
        cweb_slice = _load_global_cweb_slice(indexed, ixg)
        im = ax.imshow(cweb_slice.T, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"CWEB global slice (ix={ixg})")
        ax.set_xlabel("grid y")
        ax.set_ylabel("grid z")
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cb.set_label("Environment class")
    fig.suptitle(
        f"CWEB Global Slices (ngrid={ngrid}, threshold={threshold}, Rsmooth={rsmooth:g})",
        fontsize=12,
    )
    out_png = run_dir / "cweb_slices_global.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    args = _parse_args()

    if args.mode == "plot":
        if not args.plot_run_dir:
            raise ValueError("In --mode plot, provide --plot-run-dir.")
        _plot_existing_run(args.plot_run_dir, args.slice_fracs)
        return

    require_cpu_mpi_slurm("abacus_cactus_tweb_fullgrid_mpi.py", min_tasks=2)

    MPI = mpiutils.MPI()
    rank = MPI.rank

    density_path = Path(args.density_path)
    if not density_path.exists():
        raise FileNotFoundError(f"Density file not found: {density_path}")

    dens_local, ngrid, x_start, x_end = _load_local_density_from_full(
        density_path, args.boxsize, MPI
    )

    output_root = Path(args.output_dir)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)

    for rs in args.rsmooth:
        run_dir = _run_dir(output_root, args.backend, ngrid, rs)
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Running backend={args.backend}, Rsmooth={rs:g} ===")
        MPI.wait()

        cweb_local, eig_vals_local = _run_backend(
            args.backend,
            dens_local,
            args.boxsize,
            ngrid,
            args.threshold,
            MPI,
            rsmooth=rs,
            verbose=True,
        )

        outpath = _rank_output_path(run_dir, rank)
        np.savez(
            outpath,
            cweb=cweb_local.astype(np.uint8),
            eig_vals=eig_vals_local.astype(np.float32),
            x_start=x_start,
            x_end=x_end,
            ngrid=ngrid,
            boxsize=args.boxsize,
            threshold=args.threshold,
            Rsmooth=rs,
        )
        print(f"[rank {rank}] Saved {outpath}")
        MPI.wait()

        if rank == 0 and args.plot_slices:
            _plot_local_diagnostics(
                run_dir,
                cweb_local,
                eig_vals_local,
                ngrid,
                args.boxsize,
                args.threshold,
                rs,
            )
        if rank == 0 and args.stitch_cube:
            indexed = _index_rank_outputs(run_dir)
            _stitch_cweb_cube(run_dir, indexed)
        MPI.wait()

    MPI.end()


if __name__ == "__main__":
    main()

