import numpy as np
import matplotlib.pyplot as plt
import fitsio
import astropy.units as u
import astropy.constants as c
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo

path = '/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits'
data = Table(fitsio.read(path, columns=['RA', 'DEC', 'Z']))

# Convert RA, DEC and z to cartesian assuming plank18 cosmology

comoving_distance = cosmo.comoving_distance(data['Z']).to(u.Mpc)
sky_coord_icrs = SkyCoord(ra=data['RA'], dec=data['DEC'], unit=(u.deg, u.deg), distance=comoving_distance, frame='icrs')
cart = sky_coord_icrs.cartesian

X = cart.x.to(u.Mpc).value.astype(np.float32)
Y = cart.y.to(u.Mpc).value.astype(np.float32)
Z = cart.z.to(u.Mpc).value.astype(np.float32)

# # plot the galaxies in 2D with 0<=Z<=500 Mpc
# mask = (Z >= 0) & (Z <= 50)
# plt.figure(figsize=(10, 10))
# plt.scatter(X[mask], Y[mask], s=1, alpha=0.5)
# plt.xlabel('X (Mpc)')
# plt.ylabel('Y (Mpc)')
# plt.title(f'Galaxies in 2D ({len(data):,} galaxies)')
# plt.grid()
# plt.show()

points = np.vstack((X, Y, Z)).T

np.save("abacus_cartesian_coords.npy", points)
print("Saved cartesian coordinates to abacus_cartesian_coords.npy")