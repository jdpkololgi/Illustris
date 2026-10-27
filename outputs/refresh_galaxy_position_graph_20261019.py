from pathlib import Path
from graphify.watch import _rebuild_code
root=Path('/global/u2/d/dkololgi/TNG/Illustris')
names=['annotate_cutsky_with_tweb_eigs.py','join_cutsky_eigs_to_fiberassign_catalog.py','p10_build_bright_parent.py','p10_build_observed_truth.py','write_fiberassign_mock_science_fits.py','write_stage3_postcollision_science_fits.py','galaxy_position_mapping.py','tweb_grid_sampling.py','validate_galaxy_position_mapping.py','qualify_cubic_local_indices.py','certify_galaxy_position_scope.py','check_unresolved_satellite_hosts.py','inventory_galaxy_position_rebuilds.py','migrate_p10_galaxy_position_truth.py','smoke_galaxy_position_annotation.py','run_galaxy_position_tests.py']
changed=[root/'workflows/abacus_tweb'/x for x in names]
changed += [root/'tests/test_product_lifecycle.py',root/'workflows/abacus_tweb/build_abacus_sbi_cache.py',root/'workflows/abacus_tweb/galaxy_position_validation_status.py',root/'workflows/abacus_tweb/count_unresolved_satellites.py',root/'workflows/abacus_tweb/audit_product_lifecycle.py',root/'workflows/abacus_tweb/product_lifecycle.py',root/'tests/test_galaxy_position_annotation.py',root/'workflows/abacus_tweb/secondgen_mocks/ph000/scripts/build_mock_bgs_maglim_catalog.py']
assert _rebuild_code(root,changed_paths=changed,acquire_lock=False)
