import numpy as np
import matplotlib.pyplot as plt
import fitsio
import astropy.units as u
import astropy.constants as c
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from config_paths import ABACUS_CARTESIAN_OUTPUT, CUTSKY_Z0200_PATH

# Workflow status: ACTIVE utility (CutSky -> Cartesian coordinates)

class AbacusCatalog:
    def __init__(self, PATH):
        
        self.abcat = Table(fitsio.read(PATH))
        self.abcat = self.abcat[(self.abcat['IN_Y1'] == 1) | (self.abcat['IN_Y5'] == 1)]
        print(f"Raw number of Abacus entries: {len(self.abcat):,} in Y1 or Y5")
        
    def save_cartesian_coords(self, OUTPUT_PATH):
        """
        Convert RA, DEC, Z to Cartesian coordinates and save to OUTPUT_PATH.
        """
        # Convert RA, DEC and z to cartesian assuming Planck18 cosmology
        self.comoving_distance = cosmo.comoving_distance(self.abcat['Z']).to(u.Mpc)
        self.sky_coord_icrs = SkyCoord(ra=self.abcat['RA'], dec=self.abcat['DEC'], unit=(u.deg, u.deg), distance=self.comoving_distance, frame='icrs')
        # Convert to galactic coordinates
        self.sky_gal = self.sky_coord_icrs.galactic
        self.galactic_north_mask = self.sky_gal.b.deg > 0
        self.galactic_south_mask = ~self.galactic_north_mask
        # Get full cartesian coordinates

        cart = self.sky_coord_icrs.cartesian

        self.X = cart.x.to(u.Mpc).value.astype(np.float64)
        self.Xn = cart[self.galactic_north_mask].x.to(u.Mpc)
        self.Xs = cart[self.galactic_south_mask].x.to(u.Mpc)

        self.Y = cart.y.to(u.Mpc).value.astype(np.float64)
        self.Yn = cart[self.galactic_north_mask].y.to(u.Mpc)
        self.Ys = cart[self.galactic_south_mask].y.to(u.Mpc)

        self.Z = cart.z.to(u.Mpc).value.astype(np.float64)
        self.Zn = cart[self.galactic_north_mask].z.to(u.Mpc)
        self.Zs = cart[self.galactic_south_mask].z.to(u.Mpc)
        # flag column to indicate hemisphere. 0 = south, 1 = north
        HEMISPHERE_FLAG = self.galactic_north_mask.astype(np.int8)
        points = np.vstack((self.X, self.Y, self.Z, HEMISPHERE_FLAG)).T
        np.save(OUTPUT_PATH, points)
        print(f"Saved cartesian coordinates to {OUTPUT_PATH}")

    def plot_galaxies(self, xyzplot='hemispheres'):
        """
        Plot galaxies in 2D or 3D based on xyzplot parameter.
        """
        if xyzplot == 'hemispheres':
            X = np.concatenate((self.Xn, self.Xs))
            Y = np.concatenate((self.Yn, self.Ys))
            Z = np.concatenate((self.Zn, self.Zs))
        elif xyzplot == 'full':
            X = self.X
            Y = self.Y
            Z = self.Z
        else:
            raise ValueError("xyzplot must be either 'hemispheres' or 'full'")

        plt.figure(figsize=(10, 10))
        plt.scatter(X, Y, c=Z, s=1, alpha=0.5)
        plt.xlabel('X (Mpc)')
        plt.ylabel('Y (Mpc)')
        plt.title(f'Galaxies in 2D ({len(self.abcat):,} galaxies)')
        plt.colorbar(label='Z (Mpc)')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    path = CUTSKY_Z0200_PATH
    AbacusCatalog(path).save_cartesian_coords(ABACUS_CARTESIAN_OUTPUT)

# data = Table(fitsio.read(path, columns=['RA', 'DEC', 'Z']))

# # Convert RA, DEC and z to cartesian assuming plank18 cosmology

# comoving_distance = cosmo.comoving_distance(data['Z']).to(u.Mpc)
# sky_coord_icrs = SkyCoord(ra=data['RA'], dec=data['DEC'], unit=(u.deg, u.deg), distance=comoving_distance, frame='icrs')

# # cart = sky_coord_icrs.cartesian

# X = cart.x.to(u.Mpc).value.astype(np.float32)
# Y = cart.y.to(u.Mpc).value.astype(np.float32)
# Z = cart.z.to(u.Mpc).value.astype(np.float32)

# # plot the galaxies in 2D with 0<=Z<=500 Mpc
# mask = (Z >= 0) & (Z <= 50)
# plt.figure(figsize=(10, 10))
# plt.scatter(X[mask], Y[mask], s=1, alpha=0.5)
# plt.xlabel('X (Mpc)')
# plt.ylabel('Y (Mpc)')
# plt.title(f'Galaxies in 2D ({len(data):,} galaxies)')
# plt.grid()
# plt.show()

# plot on skymap molleweide projection
# plt.figure(figsize=(18, 6))
# ax = plt.subplot(111, projection='mollweide')
# ax.scatter(np.radians(data['RA']) - np.pi, np.radians(data['DEC']), c=data['Z'], cmap='inferno_r', s=0.1, alpha=0.5, marker='.')
# ax.set_xlabel('RA (rad)')
# ax.set_ylabel('DEC (rad)')
# ax.set_title(f'Sky Coordinates ({len(data):,} galaxies)')
# plt.colorbar(label='Redshift (z)')
# plt.grid()
# plt.show()

# points = np.vstack((X, Y, Z)).T

# np.save("abacus_cartesian_coords.npy", points)
# print("Saved cartesian coordinates to abacus_cartesian_coords.npy")