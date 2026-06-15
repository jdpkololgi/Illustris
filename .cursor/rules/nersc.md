# NERSC / Perlmutter + DESI quick reference (Cursor rules)

This rule is meant to be kept **close to the code/data** and to steer the agent toward NERSC-safe commands and conventions.

## Perlmutter gotchas (high priority)

- `--account` is effectively mandatory in Slurm at NERSC. GPU jobs often require the `_g` suffix (example: `desi_g`).
- **Use `srun` to actually run on allocated nodes**. A bare `python ...` inside an `sbatch` script can run on the first node or behave unexpectedly; prefer `srun python ...` (or `srun bash -lc '...'`).
- QOS namespaces are often split for CPU vs GPU. Don’t mix CPU QOS names with GPU QOS names.
- `$PSCRATCH` is **purged** (policy-based; don’t treat as durable).
- `$HOME` inode quotas can be the limiting factor (conda/pip caches are common offenders).
- Compute nodes may have **no outbound internet**. Avoid `pip install` at runtime; build envs on login nodes or via prebuilt containers/modules.
- `PYTHONPATH` set in shell rc files can “leak” into venv/conda and cause confusing imports. If debugging env issues, consider `unset PYTHONPATH` before activation.
- Login nodes are shared and throttled; anything heavy should be in `salloc`/`sbatch`.

## Filesystems (rule-of-thumb)

- **CFS** (`/global/cfs/...`): durable project/user storage. Prefer for inputs/outputs you want to keep.
- **PSCRATCH** (`$PSCRATCH`, typically under `/pscratch/sd/<u>/<user>`): large/fast scratch. Treat as temporary and purge-prone.
- Avoid metadata-heavy traversal on shared filesystems (e.g. `find` on huge trees). Prefer narrower globs or known directory lists.

## Slurm patterns that work well

### Interactive CPU

Prefer:

- `salloc ...` then `srun ...`
- Or `srun --pty bash -l`

### Interactive GPU

Prefer:

- `salloc -C gpu -G 1 ...` then `srun ...`

### Batch

- Use `sbatch templates/job.sbatch`
- Always set the account and a sensible QOS/partition for the resource type.

## DESI quick notes

- Prefer scripts or authoritative locations to discover “latest production” names; they drift.
- Common user output location: `/global/cfs/cdirs/desi/users/$USER/`
- If asked for file format/HDU spec, refer to the DESI data model docs.

