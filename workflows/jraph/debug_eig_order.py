import numpy as np
import os

path = '/pscratch/sd/d/dkololgi/Cosmic_env_TNG_cache/snap_099_eig_vals_0.npz'
if not os.path.exists(path):
    print("File not found: " + path)
    exit()

print("Checking " + path + "...")
data = np.load(path)
eig_vals = data['eig_vals']
print("Shape: " + str(eig_vals.shape))

# Handle potential 4D shape [3, 256, 256, 256]
if len(eig_vals.shape) == 4:
    e1 = eig_vals[0].flatten()
    e2 = eig_vals[1].flatten()
    e3 = eig_vals[2].flatten()
else:
    # If already flattened or different shape
    e1 = eig_vals[0]
    e2 = eig_vals[1]
    e3 = eig_vals[2]

n_total = len(e1)
n_samples = min(n_total, 1000000)
sample_indices = np.random.choice(n_total, n_samples, replace=False)
s1 = e1[sample_indices]
s2 = e2[sample_indices]
s3 = e3[sample_indices]

# Check sorting
ordered = np.all((s1 <= s2) & (s2 <= s3))
print("Strictly ordered (s1 <= s2 <= s3): " + str(ordered))
if not ordered:
    v1 = np.sum(s1 > s2)
    v2 = np.sum(s2 > s3)
    v3 = np.sum(s1 > s3)
    print("Ordering violations (s1 > s2): " + str(v1))
    print("Ordering violations (s2 > s3): " + str(v2))
    print("Total violations in " + str(n_samples) + " samples: " + str(v1 + v2))
    
# Stats
print("E1: min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(e1), np.max(e1), np.mean(e1)))
print("E2: min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(e2), np.max(e2), np.mean(e2)))
print("E3: min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(e3), np.max(e3), np.mean(e3)))

# Check I1 range
i1 = e1 + e2 + e3
print("I1 (trace): min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(i1), np.max(i1), np.mean(i1)))

# Check e, p ranges with current logic
# I1 trace can be zero.
denom = 2 * np.abs(i1)
e = (e3 - e1) / np.maximum(denom, 1e-8)
p = (e1 + e3 - 2*e2) / np.maximum(denom, 1e-8)

print("E (pre-clip): min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(e), np.max(e), np.mean(e)))
print("P (pre-clip): min={:.3f}, max={:.3f}, mean={:.3f}".format(np.min(p), np.max(p), np.mean(p)))

e_clipped = np.clip(e, 0, 1)
p_clipped = np.clip(p, -1, 1)

print("E (clipped): Mean={:.3f}, Std={:.3f}".format(np.mean(e_clipped), np.std(e_clipped)))
print("P (clipped): Mean={:.3f}, Std={:.3f}".format(np.mean(p_clipped), np.std(p_clipped)))
