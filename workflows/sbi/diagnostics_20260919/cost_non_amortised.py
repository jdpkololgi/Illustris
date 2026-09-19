"""Cost the non-amortised (learned prior + explicit likelihood) inference route.

Compares three ways of producing p(delta | BGS-like galaxies, selection):
  A. amortised conditional generative posterior, as proposed for Stage A/B;
  B. non-amortised MCMC with a learned unconditional score prior and an explicit
     galaxy likelihood, on the same registered panels;
  C. the same, on the reduced panel a non-amortised sampler actually needs;
  D. the survey-scale product, and a like-for-like comparison with BORG/Manticore.

Every network rate is MEASURED (GPU receipt c9eb5cd..., job 58559826, A100-SXM4-80GB,
float32, batch 2), documented in docs/e2e_coupled_resource_proposal_20260918.md.
The one genuinely unknown quantity is the number of score evaluations per
independent posterior sample, so it is swept rather than assumed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# --- measured, from the frozen GPU benchmark -------------------------------
RECT = (64, 48, 48)                 # coupled rectangle, 6.766 Mpc/h cells
RECT_VOXELS = int(np.prod(RECT))
SEC_PER_NFE = 6.6261 / 128          # J fine, s per network evaluation per field
SEC_PER_UPDATE = 0.2330             # J fine, batch-two training update
PEAK_GIB = 1.105                    # J fine peak reserved GPU memory at batch 2
AMORTISED_NFE = 128
UPDATES_PER_EPOCH = 832

# --- registered panels -----------------------------------------------------
CONFIRMATION_FIELDS = 96 * 2 * 128  # pairs x seeds x draws, four arms counted separately
DEV_FIELDS = 45056                  # proposed development panel
PROPOSED_STAGE_A_GPUH = 260
PROPOSED_STAGE_B_GPUH = 450


def comoving_volume(z_lo, z_hi, area_deg2, om=0.315, h=0.6736):
    """Flat LCDM comoving volume in (Mpc/h)^3 for a survey shell."""
    c_over_h0 = 2997.92458  # Mpc/h
    z = np.linspace(0, z_hi, 20001)
    e = np.sqrt(om * (1 + z) ** 3 + (1 - om))
    chi = c_over_h0 * np.concatenate([[0], np.cumsum(np.diff(z) / (0.5 * (e[1:] + e[:-1])))])
    lo, hi = np.interp([z_lo, z_hi], z, chi)
    sky = area_deg2 * (np.pi / 180) ** 2
    return sky / 3 * (hi ** 3 - lo ** 3)


def gpuh(seconds):
    return seconds / 3600.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evals', type=int, nargs='+', default=[2000, 5000, 20000],
                        help='score evaluations per independent posterior sample')
    parser.add_argument('--epochs', type=int, default=384, help='literature-parity training depth')
    parser.add_argument('--bgs-area', type=float, default=14000.0)
    parser.add_argument('--bgs-z', type=float, nargs=2, default=[0.1, 0.4])
    parser.add_argument('--out', type=Path,
                        default=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/diagnostics_20260919'))
    args = parser.parse_args()

    # --- training ---------------------------------------------------------
    updates = args.epochs * UPDATES_PER_EPOCH
    amortised_training = gpuh(14 * updates * SEC_PER_UPDATE)      # 14 registered factor fits
    prior_training = gpuh(2 * updates * SEC_PER_UPDATE)           # one unconditional prior, two seeds

    # --- per-field sampling ----------------------------------------------
    amortised_per_field = AMORTISED_NFE * SEC_PER_NFE
    result = dict(
        schema='e2e-cost-non-amortised-v1',
        measured=dict(seconds_per_network_evaluation=SEC_PER_NFE,
                      seconds_per_training_update=SEC_PER_UPDATE,
                      rectangle_voxels=RECT_VOXELS, peak_gib_batch2=PEAK_GIB,
                      source='GPU receipt c9eb5cd..., job 58559826'),
        training=dict(epochs=args.epochs, updates_per_factor=updates,
                      amortised_14_factors_gpuh=amortised_training,
                      unconditional_prior_2_seeds_gpuh=prior_training),
        options={})

    # A. amortised, as proposed
    result['options']['A_amortised_as_proposed'] = dict(
        description='14 conditional factors; development + 128-draw confirmation',
        training_gpuh=amortised_training,
        confirmation_fields=CONFIRMATION_FIELDS,
        confirmation_gpuh=gpuh(CONFIRMATION_FIELDS * amortised_per_field),
        development_gpuh=gpuh(DEV_FIELDS * amortised_per_field),
        proposed_ceiling_gpuh=PROPOSED_STAGE_A_GPUH + PROPOSED_STAGE_B_GPUH)

    # B / C. non-amortised on the registered panel, and on a reduced one
    for name, fields, note in (
            ('B_non_amortised_same_panel', CONFIRMATION_FIELDS,
             'same confirmation panel as the amortised plan'),
            ('C_non_amortised_reduced_panel', 24 * 64,
             '24 fields x 64 draws: a non-amortised sampler validates its own calibration, '
             'so the panel exists to check the sampler and the likelihood, not to check '
             'generalisation across configurations')):
        rows = {}
        for n in args.evals:
            rows[str(n)] = dict(
                sampling_gpuh=gpuh(fields * n * SEC_PER_NFE),
                total_gpuh=prior_training + gpuh(fields * n * SEC_PER_NFE),
                cost_multiple_vs_amortised=(n / AMORTISED_NFE))
        result['options'][name] = dict(description=note, fields=fields,
                                       training_gpuh=prior_training, by_evaluations=rows)

    # D. survey-scale product
    volume = comoving_volume(args.bgs_z[0], args.bgs_z[1], args.bgs_area)
    voxels = volume / 6.766 ** 3
    scale = voxels / RECT_VOXELS
    memory_gib = PEAK_GIB / 2 * scale          # per field, from the batch-2 peak
    sec_per_eval = SEC_PER_NFE * scale
    bottleneck_tokens = voxels / 8 ** 3        # three U-Net downsamplings
    survey = dict(
        area_deg2=args.bgs_area, redshift_range=args.bgs_z,
        comoving_volume_gpc_h3=volume / 1e9, voxels_at_6p766_mpc_h=voxels,
        rectangles_equivalent=scale,
        single_field_memory_gib=memory_gib,
        seconds_per_score_evaluation=sec_per_eval,
        global_attention_tokens=bottleneck_tokens,
        global_attention_pair_count=bottleneck_tokens ** 2,
        by_evaluations={str(n): dict(
            gpuh_per_posterior_sample=gpuh(n * sec_per_eval),
            gpuh_for_100_samples=gpuh(100 * n * sec_per_eval),
            gpu_node_hours_for_100_samples=gpuh(100 * n * sec_per_eval) / 4)
            for n in args.evals})
    # Manticore II reference: ~30M CPU-hours, 128 cores per Perlmutter CPU node
    survey['manticore_reference'] = dict(
        cpu_hours=30e6, cpu_node_hours=30e6 / 128,
        note='order-of-magnitude reference only; different survey, different product '
             '(initial conditions and joint cosmology), different machine')
    for n in args.evals:
        node_hours = survey['by_evaluations'][str(n)]['gpu_node_hours_for_100_samples']
        survey['by_evaluations'][str(n)]['ratio_manticore_node_hours_over_this'] = (
            (30e6 / 128) / node_hours if node_hours else None)
    result['options']['D_survey_scale_product'] = survey

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'COST_NON_AMORTISED.json'
    path.write_text(json.dumps(result, indent=1, sort_keys=True))

    a = result['options']['A_amortised_as_proposed']
    print(f"measured: {SEC_PER_NFE*1000:.2f} ms per score evaluation on {RECT_VOXELS:,} voxels\n")
    print(f"TRAINING at {args.epochs} epochs: 14 amortised factors {amortised_training:6.1f} GPUh"
          f"   |  unconditional prior x2 seeds {prior_training:5.1f} GPUh")
    print(f"\nA. amortised as proposed: train {a['training_gpuh']:.1f} + dev "
          f"{a['development_gpuh']:.1f} + confirmation {a['confirmation_gpuh']:.1f} GPUh"
          f"   (proposed ceilings {a['proposed_ceiling_gpuh']})")
    for name in ('B_non_amortised_same_panel', 'C_non_amortised_reduced_panel'):
        o = result['options'][name]
        print(f"\n{name}  ({o['fields']:,} fields)")
        for n in args.evals:
            r = o['by_evaluations'][str(n)]
            print(f"   {n:>6,} evals/sample: sampling {r['sampling_gpuh']:9.1f} GPUh"
                  f"   total {r['total_gpuh']:9.1f} GPUh   ({r['cost_multiple_vs_amortised']:.0f}x per draw)")
    s = result['options']['D_survey_scale_product']
    print(f"\nD. BGS {args.bgs_area:.0f} deg^2, z {args.bgs_z[0]}-{args.bgs_z[1]}: "
          f"{s['comoving_volume_gpc_h3']:.2f} (Gpc/h)^3, {s['voxels_at_6p766_mpc_h']/1e6:.1f}M voxels "
          f"= {s['rectangles_equivalent']:.0f} rectangles")
    print(f"   one field fits in {s['single_field_memory_gib']:.1f} GiB; "
          f"{s['seconds_per_score_evaluation']:.2f} s per global score evaluation")
    print(f"   global attention would need {s['global_attention_tokens']:,.0f} tokens "
          f"({s['global_attention_pair_count']:.2e} pairs)")
    for n in args.evals:
        r = s['by_evaluations'][str(n)]
        print(f"   {n:>6,} evals: {r['gpuh_per_posterior_sample']:8.1f} GPUh per sample, "
              f"{r['gpu_node_hours_for_100_samples']:9.0f} GPU node-h for 100 samples"
              f"   ({r['ratio_manticore_node_hours_over_this']:.0f}x cheaper than the Manticore reference)")
    print('\nWROTE', path)


if __name__ == '__main__':
    main()
