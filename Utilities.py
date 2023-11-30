import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def readsnap(path, snapno, xyzplot=True, lim=5000):
    '''Reads the snapshot data from the given path and snapshot number.'''
    object = il.groupcat.load(path,snapno)

    if xyzplot:
        sf = object['header']['Time'] #time of snapshot
        hub = object['header']['HubbleParam'] #hubble parameter of simulation

        X = object['halos']['GroupPos'][:,0][:lim]*sf/hub #spatial coordinates of halos
        Y = object['halos']['GroupPos'][:,1][:lim]*sf/hub
        Z = object['halos']['GroupPos'][:,2][:lim]*sf/hub

        x = object['subhalos']['SubhaloPos'][:,0][:lim]*sf/hub #spatial coordinates of subhalos
        y = object['subhalos']['SubhaloPos'][:,1][:lim]*sf/hub
        z = object['subhalos']['SubhaloPos'][:,2][:lim]*sf/hub
        sfr = object['subhalos']['SubhaloSFRinRad'][:lim]

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        ax.grid(True)

        ax.scatter3D(x, y, z, c=sfr, cmap='viridis', s=4)
        ax.scatter3D(X, Y, Z, s=10)
        ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='z', style='sci',scilimits=(0,0))
        ax.set_xlabel(r'x [$\times10^{2}\,Mpc$]')
        ax.set_ylabel(r'y [$\times10^{2}\,Mpc$]')
        ax.set_zlabel(r'z [$\times10^{2}\,Mpc$]')
        plt.show()

    return object

if __name__ == '__main__':
    readsnap(r'/Users/daksheshkololgi/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 1/Illustris/TNG300-1', 99)