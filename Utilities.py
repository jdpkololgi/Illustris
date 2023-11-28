import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt

def readsnap(path, snapno, xyzplot=True, lim=500, colorbar=True):
    '''Reads the snapshot data from the given path and snapshot number.'''
    object = il.groupcat.load(path,snapno)

    if xyzplot:
        X = object['halos']['GroupPos'][:,0][:lim] #spatial coordinates of halos
        Y = object['halos']['GroupPos'][:,1][:lim]
        Z = object['halos']['GroupPos'][:,2][:lim]

        x = object['subhalos']['SubhaloPos'][:,0][:lim] #spatial coordinates of subhalos
        y = object['subhalos']['SubhaloPos'][:,1][:lim]
        z = object['subhalos']['SubhaloPos'][:,2][:lim]
        sfr = object['subhalos']['SubhaloSFRinRad'][:lim]

        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        ax.grid(True)

        ax.scatter3D(x, y, z, c=sfr, cmap='viridis', s=4)
        ax.scatter3D(X, Y, Z, s=10)
        fig.colorbar(ax.scatter3D(x, y, z, c=np.log(sfr), cmap='viridis', s=4), ax=ax)
        plt.show()

    return object