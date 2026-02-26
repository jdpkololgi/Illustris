# TNG/Illustris Runbook

This runbook lists validated workflow entrypoints and launch commands.

## Environment setup

```bash
source ~/.bashrc
conda activate cosmic_env
desienv
```

## Canonical workflows

### Abacus slab + MPI T-Web (CPU)

Batch launch:

```bash
sbatch workflows/abacus_tweb/submit_abacus_tweb_cpu.slurm
```

Direct entrypoints:

```bash
python workflows/abacus_tweb/abacus_cactus_tweb.py
python workflows/abacus_tweb/annotate_cutsky_with_tweb.py
python workflows/abacus_tweb/abacus_process_particles2.py --show-workflow
```

### Jraph training baseline (GPU)

Batch launch:

```bash
sbatch workflows/jraph/submit_jraph.slurm
```

Other launchers:

```bash
sbatch workflows/jraph/debug_jraph.slurm
sbatch workflows/jraph/submit_tuning.slurm
sbatch workflows/jraph/train_ensemble.slurm
```

Direct:

```bash
python workflows/jraph/jraph_pipeline.py --help
```

### SBI FlowJAX canonical path (GPU)

Batch launch:

```bash
sbatch workflows/sbi/submit_sbi_flowjax.slurm
```

Direct:

```bash
python workflows/sbi/jraph_sbi_flowjax.py --help
```

### Optional experimental SBI two-stage path (GPU)

Batch launch:

```bash
sbatch workflows/sbi/experimental/run_sbi_two_stage.slurm
```

Direct:

```bash
python workflows/sbi/experimental/jraph_sbi_two_stage.py --help
```

### GCN paper workflow (GPU)

Batch launch:

```bash
sbatch workflows/gcn_paper/submit_gcn.slurm
```

Direct:

```bash
python workflows/gcn_paper/gcn_pipeline.py --help
python workflows/gcn_paper/gcn_pipeline_postprocess.py --help
python workflows/gcn_paper/postprocessing.py --help
```

## Compatibility wrappers (temporary)

Root-level script names still forward to canonical or legacy modules during migration.
Prefer canonical `workflows/...` paths for all new scripts and docs.

Examples:

```bash
python jraph_pipeline.py --help
python jraph_sbi_flowjax.py --help
python jraph_sbi_pipeline.py --help
```

## Notes

- Active workflows and statuses: `ACTIVE_WORKFLOWS.md`
- Migration docs: `docs/migration/`
- Legacy scripts for reference: `legacy/`
