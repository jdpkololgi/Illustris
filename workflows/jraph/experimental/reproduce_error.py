
import numpy as np
import pickle
import time
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Allow canonical workflow scripts to resolve repo-root modules after reorganization.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eigenvalue_transformations as et


print("Loading real data...")
with open('/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/processed_jraph_data_mc1e+09_v2_scaled_3_eigenvalues.pkl', 'rb') as f:
    data = pickle.load(f)
    raw_eig = data.get('eigenvalues_raw')
    standardised_eig = data.get('regression_targets')

print(f"Loaded {len(raw_eig)} samples.")
eigenvalues_in = jnp.array(raw_eig, dtype=jnp.float32)
standardised_eig = np.array(standardised_eig)

# # Shape Params transform
# print("Testing Shape Params...")
# params = et.eigenvalues_to_shape_params(eigenvalues_in)
# eigenvalues_out_shape = et.shape_params_to_eigenvalues(params)

# errs_shape = jnp.max(jnp.abs(eigenvalues_in - eigenvalues_out_shape), axis=1)
# max_err_shape = jnp.max(errs_shape)
# mse_shape = jnp.mean((eigenvalues_in - eigenvalues_out_shape)**2)

# print(f"Shape MSE: {mse_shape:.6e}")
# print(f"Shape Max Error: {max_err_shape:.6e}")

# # Filter out points where I1 < 0.01 (where forward transform masks anisotropy)
# valid_mask = jnp.abs(jnp.sum(eigenvalues_in, axis=1)) > 1e-2
# print(f"Number of masked samples (|I1| < 0.01): {jnp.sum(~valid_mask)}")

# errs_shape_valid = errs_shape[valid_mask]
# max_err_shape_valid = jnp.max(errs_shape_valid)
# mse_shape_valid = jnp.mean(errs_shape_valid**2)

# print(f"Shape MSE (Valid Only): {mse_shape_valid:.6e}")
# print(f"Shape Max Error (Valid Only): {max_err_shape_valid:.6e}")

# print("-" * 20)
print("Testing Invariants...")
# Invariants transform
invariants = et.eigenvalues_to_invariants(eigenvalues_in)
eigenvalues_out_inv = et.invariants_to_eigenvalues(invariants)
print('Invariants transform complete.')
# Compute absolute differences
diffs = jnp.abs(eigenvalues_in - eigenvalues_out_inv)
# Find max error per sample for sorting/indexing
errs_per_sample = jnp.max(diffs, axis=1)

max_err_inv = jnp.max(diffs)
mse_inv = jnp.mean((eigenvalues_in - eigenvalues_out_inv)**2)

print(f"MSE: {mse_inv:.6e}")
print(f"Max Error: {max_err_inv:.6e}")

# Find index of max error
idx = jnp.argmax(errs_per_sample)
print(f"Worst case index: {idx}")
print(f"Worst case inputs: {eigenvalues_in[idx]}")
print(f"Worst case outputs: {eigenvalues_out_inv[idx]}")
print(f"Worst case error vector: {diffs[idx]}")

# Check for failure cases (> 1e-4)
failures = jnp.where(errs_per_sample > 1e-4)[0]
print(f"Number of failures (>1e-4): {len(failures)}")
if len(failures) > 0:
    print("Example failures:")
    for i in failures[:5]:
        print(f"Index {i}: In={eigenvalues_in[i]}, Out={eigenvalues_out_inv[i]}, Err={diffs[i]}")

# Shape Parameters
params = et.eigenvalues_to_shape_params(eigenvalues_in)
params_np = np.array(params)

# Invariants
inv_np = np.array(invariants)

###########################
# TEST CUMULATIVE SOFTPLUS
###########################
print("\nTesting Cumulative Softplus Increments...")
t0 = time.time()
increments = et.eigenvalues_to_increments(eigenvalues_in)
eigenvalues_out_inc = et.increments_to_eigenvalues(increments)
jax.block_until_ready(eigenvalues_out_inc)
print(f"Cumulative Softplus transform complete in {time.time() - t0:.4f}s")

# MSE and Max Error
diffs_inc = jnp.abs(eigenvalues_in - eigenvalues_out_inc)
mse_inc = jnp.mean(diffs_inc**2)
max_err_inc = jnp.max(diffs_inc)

print(f"MSE: {mse_inc:e}")
print(f"Max Error: {max_err_inc:e}")

# Apply PowerTransformer (Log-like scaling)
print("\nApplying PowerTransformer (Yeo-Johnson)...")
eig_params = PowerTransformer(method='yeo-johnson')
eig_scaled = eig_params.fit_transform(np.array(eigenvalues_in))

pt_params = PowerTransformer(method='yeo-johnson')
params_scaled = pt_params.fit_transform(params_np)

pt_inv = PowerTransformer(method='yeo-johnson')
inv_scaled = pt_inv.fit_transform(inv_np)

# Plotting Histograms
print("Plotting histograms...")
pass # just to cleanly separate blocks
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
fig.suptitle("Distributions: Raw vs PowerTransformer (Log-Scaled)")

# Row 1: Raw Eigenvalues
eig_np = np.array(eigenvalues_in)
axes[0, 0].hist(eig_np[:, 0], bins=500, alpha=0.7)
axes[0, 0].set_title(r"Raw $\lambda_1$")
axes[0, 1].hist(eig_np[:, 1], bins=500, alpha=0.7)
axes[0, 1].set_title(r"Raw $\lambda_2$")
axes[0, 2].hist(eig_np[:, 2], bins=500, alpha=0.7)
axes[0, 2].set_title(r"Raw $\lambda_3$")

# Row 2: Raw Shape Parameters
axes[1, 0].hist(params_np[:, 0], bins=100, alpha=0.7, color='green')
axes[1, 0].set_title(r"Raw $I_1$ (Trace)")
axes[1, 1].hist(params_np[:, 1], bins=100, alpha=0.7, color='green')
axes[1, 1].set_title(r"Raw $e$ (Ellipticity)")
axes[1, 2].hist(params_np[:, 2], bins=100, alpha=0.7, color='green')
axes[1, 2].set_title(r"Raw $p$ (Prolateness)")

# Row 3: Softplus distributions
increments = np.array(increments)
axes[2, 0].hist(increments[:, 0], bins=500, alpha=0.7, color='purple')
axes[2, 0].set_title(r"Scaled $v_1$")
axes[2, 1].hist(increments[:, 1], bins=500, alpha=0.7, color='purple')
axes[2, 1].set_title(r"Scaled $v_2$")
axes[2, 2].hist(increments[:, 2], bins=500, alpha=0.7, color='purple')
axes[2, 2].set_title(r"Scaled $v_3$")

# Row 4: Scaled Invariants (PowerTransformer)
axes[3, 0].hist(inv_scaled[:, 0], bins=100, alpha=0.7, color='red')
axes[3, 0].set_title(r"Scaled $I_1$ (Gaussianized)")
axes[3, 1].hist(inv_scaled[:, 1], bins=100, alpha=0.7, color='red')
axes[3, 1].set_title(r"Scaled $I_2$ (Gaussianized)")
axes[3, 2].hist(inv_scaled[:, 2], bins=100, alpha=0.7, color='red')
axes[3, 2].set_title(r"Scaled $I_3$ (Gaussianized)")

plt.tight_layout()
plt.savefig("target_distributions_scaled3.png", dpi=100)
print("Saved histogram plot to target_distributions_scaled3.png")
