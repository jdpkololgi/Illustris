# D2 evaluation host-memory recovery, 2026-09-06

The native-support evaluation job 57987709 failed after 21m54s; its step
57987709.0 was OUT_OF_MEMORY (0:125), MaxRSS 58,737,588 KiB against an
allocation of 57,472 MiB. This is host-memory exhaustion, not evidence of
a scientific gate failure. Common, shear, spectral/TARP and three-point
artifacts were written before termination; there was no terminal freeze.

The complete partial directory was renamed from `reports_native_support_v1`
to `reports_native_support_v1_oom57987709` within the seed42 NFE50 archive.
No output was deleted and no partial result was accepted as complete.

The existing frozen launcher and evaluator are unchanged. One bounded retry
requests 2 GPUs, 64 logical CPUs and 110 GiB host memory in shared QOS;
the existing srun still computes on 1 GPU/32 logical CPUs. The additional
allocation supplies host-memory headroom for candidate and reference panels.
The wall-time limit stays 2 hours. Explicit `--gpus=2 --gpus-per-node=2
--gpus-per-task=2` flags were required by scheduler validation; an initial
submission with only per-task override was rejected and created no job.

Replacement chain:

- 57988874: native NFE50 evaluation memory retry.
- 57988878: native NFE100 evaluation, afterok 57985431 and 57988874.
- 57988884: unchanged primary scientific decision, afterok 57988878.

Stranded prior evaluation100 57987713 and decision 57987720 were cancelled
only after replacements were submitted. Export100 recovery 57985431 stays
running and was at 219/256 cores when the replacement receipt was checked.
No further automatic memory escalation is authorized by this retry record.

The partial common artifact verifies 256 cores and all 133,698 galaxies,
including 736 nearest-voxel M=0 positions with zero native-unsupported rows.
Its preliminary maximum deployable conditional errors are 0.065275906
(voxel) and 0.091977077 (derived). These are not a final D2 decision.
The native amendment SHA256 remains
3a6ef4d38c0cc659f87cef8894f57cb4b3ea1028728e094a1da2d432eee7a546.

NERSC policy reference: https://docs.nersc.gov/jobs/policy/ (shared GPU
allocations pair one/two GPUs with the corresponding CPU and RAM share).
