import numpy as np
import os
import sys
import cactus
import matplotlib.pyplot as plt
# Ensure we can import the local modules
sys.path.append(os.getcwd())

# Assuming cactus and shift are installed or in path
try:
    from cactus.ext import fiesta
    import shift
except ImportError as e:
    print(f"Warning: Could not import cactus or shift: {e}") 
    sys.exit(1)

# import TNG.Illustris.hdf5_helper as hdf5_helper

# Fixed path handling
base_path = '/pscratch/sd/d/dkololgi/TNG300-1'
if not os.path.exists(base_path):
    print(f"Warning: Path {base_path} does not exist on this machine.")

boxsize = 205.
ngrid = 512

# import density field from pscratch

fname = '/pscratch/sd/d/dkololgi/IllustrisTNG300_densities/snap_099_total_dens_0.npz'

data = np.load(fname)

dens = data['dens']
# plot log densities
# %%
fig, axes = plt.subplots(figsize=(8, 8))

axes.imshow(np.log10(dens[0]).T, origin='lower')


axes.set_xticks([])
axes.set_yticks([])

plt.tight_layout()
plt.show()

# %%
# Running T-Web
threshold = 0.2
Rsmooth = 2.
boundary = 'periodic'

cweb, eig_vals = cactus.src.tweb.run_tweb(
    dens, boxsize, ngrid, threshold, Rsmooth=Rsmooth, boundary=boundary, verbose=True)

plt.figure(figsize=(7,7))
plt.imshow(cweb[10].T, origin='lower')
plt.show()

# %%
# Save eigenvalues as npz file
# np.savez('snap_099_eig_vals_0.npz', eig_vals=eig_vals)

