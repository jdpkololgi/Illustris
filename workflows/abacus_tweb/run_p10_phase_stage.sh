#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHASE {p1|graph-cap|p2-post|p3|p3-p4} [CAP]" >&2
  exit 2
fi
phase=$1
stage=$2
cap=${3:-}
case "$phase" in ph001|ph002|ph003|ph004|ph005) ;; *) echo "unsupported phase: $phase" >&2; exit 2;; esac

repo=/global/homes/d/dkololgi/TNG/Illustris
root=/pscratch/sd/d/dkololgi/abacus/p10_multiphase
phase_root="$root/$phase"
cosmic=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
rapids=/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn/bin/python
registry="$repo/configs/p10_phase_registry_v1.json"
prefix="${phase}_bgs_bright_full_delaunay"
mock=$((10#${phase#ph}))

unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
cd "$repo"
mkdir -p "$root/logs"

run_p1() {
  if [[ "$phase" == ph001 ]]; then
    parent="$phase_root/catalogues/blind_parent/${phase}_bgs_bright_parent_linkage.fits"
    observed="$phase_root/catalogues/blind_observed/${phase}_bgs_bright_full_observed_geometry.fits"
    if [[ ! -s "$parent.complete.json" ]]; then
      "$cosmic" -u workflows/abacus_tweb/p10_build_bright_parent.py \
        --registry "$registry" --phase "$phase" --blind-geometry-only
    fi
    if [[ ! -s "$observed.complete.json" ]]; then
      "$cosmic" -u workflows/abacus_tweb/p10_build_blind_observed_geometry.py \
        --registry "$registry" --phase "$phase"
    fi
    if [[ ! -s "$phase_root/p1_canonical/CATALOGUE_COMPLETE.json" ]]; then
      "$cosmic" -u workflows/abacus_tweb/p10_build_phase_index.py \
        --registry "$registry" --phase "$phase" --blind-geometry-only
    elif [[ ! -s "$phase_root/p1_canonical/manifest.json" ]]; then
      "$cosmic" -u workflows/abacus_tweb/p10_build_phase_index.py \
        --registry "$registry" --phase "$phase" --blind-geometry-only --reuse-validated
    fi
    return
  fi

  density="$phase_root/targets/density/AbacusSummit_base_c000_${phase}_z0.200_ngrid2048_ab10_tsc_counts.npy"
  tweb="$phase_root/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json"
  [[ -s "$density" ]] || { echo "missing density: $density" >&2; exit 2; }
  [[ -s "$tweb" ]] || { echo "missing T-web marker: $tweb" >&2; exit 2; }
  parent="$phase_root/catalogues/bright_parent/${phase}_bgs_bright_parent_linkage.fits"
  annotated="$phase_root/catalogues/annotated_parent/${phase}_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits"
  observed="$phase_root/catalogues/observed/${phase}_bgs_bright_full_observed_with_tweb.fits"
  if [[ ! -s "$parent.complete.json" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p10_build_bright_parent.py \
      --registry "$registry" --phase "$phase"
  fi
  if [[ ! -s "$annotated" ]]; then
    "$cosmic" -u workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py \
      --cutsky "$parent" \
      --tweb-dir "$phase_root/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7" \
      --halo-info-dir "/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/AbacusSummit_base_c000_${phase}/halos/z0.200/halo_info" \
      --output-dir "$phase_root/catalogues/annotated_parent" \
      --output-name "$(basename "$annotated")" \
      --temp-dir "$phase_root/catalogues/annotated_parent/tmp_annotation"
  fi
  if [[ ! -s "$observed.complete.json" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p10_build_observed_truth.py \
      --registry "$registry" --phase "$phase"
  fi
  if [[ ! -s "$phase_root/p1_canonical/CATALOGUE_COMPLETE.json" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p10_build_phase_index.py \
      --registry "$registry" --phase "$phase"
  elif [[ ! -s "$phase_root/p1_canonical/manifest.json" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p10_build_phase_index.py \
      --registry "$registry" --phase "$phase" --reuse-validated
  fi
}

run_graph_cap() {
  case "$cap" in NGC|SGC) ;; *) echo "graph-cap requires NGC or SGC" >&2; exit 2;; esac
  "$cosmic" -u workflows/abacus_tweb/p10_build_phase_graph.py \
    --registry "$registry" --phase "$phase" --stage cap --cap "$cap"
}

run_p2_post() {
  "$cosmic" -u workflows/abacus_tweb/p10_build_phase_graph.py \
    --registry "$registry" --phase "$phase" --stage merge
  if [[ ! -s "$phase_root/p2_graph/${prefix}_cugraph_gnn_metadata.json" ]]; then
    CONDA_PREFIX=/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn \
    PATH=/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn/bin:"$PATH" \
    "$rapids" -u workflows/abacus_tweb/abacus_graph_features_cugraph.py \
      --artifacts-dir "$phase_root/p2_graph" --prefix "$prefix" \
      --output-dir "$phase_root/p2_graph" --output-prefix "${prefix}_cugraph"
  fi
  if [[ ! -s "$phase_root/p2_union/UNION_COMPLETE" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p2b_build_full_radius_union.py \
      --graph-dir "$phase_root/p2_graph" --prefix "$prefix" \
      --canonical-index "$phase_root/p1_canonical/canonical_index.npz" \
      --out-dir "$phase_root/p2_union" --radius-mpc 14.78
  fi
  "$cosmic" -u workflows/abacus_tweb/p10_validate_phase_products.py \
    --registry "$registry" --phase "$phase" --stage p2
}

run_p3() {
  "$cosmic" -u workflows/abacus_tweb/p10_materialize_phase_schemas.py \
    --phase "$phase" --phase-root "$phase_root"
  p3_schema="$phase_root/contracts/p3_schema_v1.json"
  if [[ ! -s "$phase_root/p3_fields/FIELD_COMPLETE" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p3a_build_canonical_fields.py \
      --points "$phase_root/p1_canonical/points.npy" \
      --canonical-index "$phase_root/p1_canonical/canonical_index.npz" \
      --p1-manifest "$phase_root/p1_canonical/manifest.json" \
      --schema "$p3_schema" \
      --ntilde-spline /pscratch/sd/d/dkololgi/abacus/conditioning/ntilde_spline_v1_frozen.json \
      --unit-audit /pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json \
      --out-dir "$phase_root/p3_fields"
  fi
}

run_p3_p4() {
  [[ -s "$phase_root/p2_graph/P2_COMPLETE.json" ]] || { echo "P2 incomplete" >&2; exit 2; }
  run_p3
  p4_schema="$phase_root/contracts/p4_schema_v1.json"
  if [[ ! -s "$phase_root/p4_patches/PATCH_MANIFEST_COMPLETE" ]]; then
    mkdir -p "$phase_root/p4_patches" "$phase_root/p4_rebuild"
    "$cosmic" -u workflows/abacus_tweb/p4_probe_core_sizes.py \
      --schema "$p4_schema" --points "$phase_root/p1_canonical/points.npy" \
      --index "$phase_root/p1_canonical/canonical_index.npz" \
      --p3-manifest "$phase_root/p3_fields/field_manifest.json" \
      --p3-unit-audit /pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json \
      --p2-manifest "$phase_root/p2_union/p2b_union_manifest.json" \
      --out "$phase_root/p4_patches/core_size_probe.json"
    if [[ "$phase" == ph001 ]]; then
      catalogue="$phase_root/catalogues/blind_observed/${phase}_bgs_bright_full_observed_geometry.fits"
    else
      catalogue="$phase_root/catalogues/observed/${phase}_bgs_bright_full_observed_with_tweb.fits"
    fi
    for out in "$phase_root/p4_patches" "$phase_root/p4_rebuild"; do
      "$cosmic" -u workflows/abacus_tweb/p4_build_spatial_manifest.py \
        --schema "$p4_schema" --probe "$phase_root/p4_patches/core_size_probe.json" \
        --points "$phase_root/p1_canonical/points.npy" \
        --index "$phase_root/p1_canonical/canonical_index.npz" --catalogue "$catalogue" \
        --p1-manifest "$phase_root/p1_canonical/manifest.json" \
        --p2-manifest "$phase_root/p2_union/p2b_union_manifest.json" \
        --p3-manifest "$phase_root/p3_fields/field_manifest.json" --out-dir "$out"
    done
    "$cosmic" -u workflows/abacus_tweb/p4_attach_graph_support.py \
      --canonical-index "$phase_root/p1_canonical/canonical_index.npz" \
      --context-assignment "$phase_root/p4_patches/context_assignment.npz" \
      --active-assignment "$phase_root/p4_patches/active_assignment.npz" \
      --p4-manifest "$phase_root/p4_patches/spatial_manifest.json" \
      --p2-manifest "$phase_root/p2_union/p2b_union_manifest.json" \
      --delaunay "$phase_root/p2_graph/${prefix}_edges_combined_idx.npy" \
      --radius-ngc "$phase_root/p2_union/ngc_radius_only_pairs.npy" \
      --radius-sgc "$phase_root/p2_union/sgc_radius_only_pairs.npy" \
      --max-k 4 --out-dir "$phase_root/p4_patches"
    "$cosmic" -u workflows/abacus_tweb/p4_attach_field_support.py \
      --points "$phase_root/p1_canonical/points.npy" \
      --p3-manifest "$phase_root/p3_fields/field_manifest.json" \
      --p4-manifest "$phase_root/p4_patches/spatial_manifest.json" \
      --active-assignment "$phase_root/p4_patches/active_assignment.npz" \
      --cores "$phase_root/p4_patches/cores.npz" --out-dir "$phase_root/p4_patches"
    "$cosmic" -u workflows/abacus_tweb/p4_finalize_validate.py \
      --geometry "$phase_root/p4_patches/spatial_manifest.json" \
      --graph-support "$phase_root/p4_patches/graph_support_manifest.json" \
      --field-support "$phase_root/p4_patches/field_support_manifest.json" \
      --rebuild "$phase_root/p4_rebuild/spatial_manifest.json" \
      --out "$phase_root/p4_patches/p4_validation.json" \
      --marker "$phase_root/p4_patches/PATCH_MANIFEST_COMPLETE"
  fi
  if [[ "$phase" != ph001 && ! -s "$phase_root/p3_fields/catalogue_field_target_closure.json" ]]; then
    "$cosmic" -u workflows/abacus_tweb/p3a_catalogue_field_closure.py \
      --points "$phase_root/p1_canonical/points.npy" \
      --index "$phase_root/p1_canonical/canonical_index.npz" \
      --catalogue "$phase_root/catalogues/observed/${phase}_bgs_bright_full_observed_with_tweb.fits" \
      --field-manifest "$phase_root/p3_fields/field_manifest.json" \
      --out "$phase_root/p3_fields/catalogue_field_target_closure.json"
  fi
  "$cosmic" -u workflows/abacus_tweb/p10_validate_phase_products.py \
    --registry "$registry" --phase "$phase" --stage phase
}

case "$stage" in
  p1) run_p1 ;;
  graph-cap) run_graph_cap ;;
  p2-post) run_p2_post ;;
  p3) run_p3 ;;
  p3-p4) run_p3_p4 ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
