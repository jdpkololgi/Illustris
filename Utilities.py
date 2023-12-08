import illustris_python as il
import mistree as mist
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import scienceplots
plt.style.use('science')

def readsnap(path, snapno, xyzplot=True, lim=5000):
    '''Reads the snapshot data from the given path and snapshot number.'''
    object = il.groupcat.load(path,snapno)

    if xyzplot:
        '''Plots in 3D the spatial coordinates of 5000 halos, finds all associated subhalos and plots them in 3D.'''
        sf = object['header']['Time'] #time of snapshot
        hub = object['header']['HubbleParam'] #hubble parameter of simulation

        X = u.kpc*object['halos']['GroupPos'][:,0][:lim]*sf/hub #spatial coordinates of halos
        Y = u.kpc*object['halos']['GroupPos'][:,1][:lim]*sf/hub
        Z = u.kpc*object['halos']['GroupPos'][:,2][:lim]*sf/hub
        groupids = np.array(object['subhalos']['SubhaloGrNr'])

        assign = []
        for i in np.arange(lim):
            assign.append(np.where(groupids==i))

        x = u.kpc*object['subhalos']['SubhaloPos'][:,0]*sf/hub #spatial coordinates of subhalos
        y = u.kpc*object['subhalos']['SubhaloPos'][:,1]*sf/hub
        z = u.kpc*object['subhalos']['SubhaloPos'][:,2]*sf/hub

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        ax.grid(True)
        # ax.scatter(x, y, z)
        ax.scatter(X.to('Mpc'), Y.to('Mpc'), Z.to('Mpc'), marker='o', color='b',s = 10, alpha=0.5, label=f'{lim} Halos')

        for i in range(len(assign)):
            ax.scatter(x[assign[i]].to('Mpc'), y[assign[i]].to('Mpc'), z[assign[i]].to('Mpc'), marker='x', color='r', label='Subhalos')
            
        # ax.scatter3D(x, y, z)#, c=sfr, cmap='viridis', s=4)
        # ax.scatter3D(X, Y, Z)
        ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='z', style='sci',scilimits=(0,0))
        ax.set_xlabel(r'x [Mpc]')
        ax.set_ylabel(r'y [Mpc]')
        ax.set_zlabel(r'z [Mpc]')
        fig.legend()
        plt.show()

    return object

def MST():
    return 0


if __name__ == '__main__':
    test=readsnap(r'/Users/daksheshkololgi/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 1/Illustris/TNG300-1', 99)