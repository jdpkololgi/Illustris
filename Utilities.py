import illustris_python as il
import mistree as mist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import astropy.constants as c
import scienceplots
plt.style.use(['science','dark_background','no-latex'])

class cat():
    def __init__(self, path, snapno):
        '''Class to initialse a group object with integrated plotting functions.
        Parameters
        ----------
        path: string
            path to pass to readsnap function.
        snapno: int
            Illustris snapshot number.

        Returns
        -------
        cat() object which will be populated with illustris group catalogue properties.
        '''
        if isinstance(path, str):
            self.path = path
        else:
            raise TypeError('Name must be a string.')
        
        if isinstance(snapno, int):
            if len(f'{snapno}') and 0<= snapno <=99:
                self.snapno = snapno
            else:
                raise ValueError(f'The snapshot number {snapno} is invalid. Please enter a number from 0 to 99.')
        else:
            raise TypeError('The snapshot number must be an integer.')
        
    def __repr__(self):
        assert hasattr(self, 'object'), 'No TNG object has been read in. Please add one using the readcat() method.'
        boxsize = round(((u.kpc*self.object['header']['BoxSize']*self.sf/self.hub).to('Mpc')).value)
        redshift = round(self.object['header']['Redshift'])
        return f'TNG{boxsize}Mpc_z={redshift}_Snapshot={self.snapno}'

    def readcat(self, xyzplot, lim=5000):
        '''Reads in groupcat data from the given path and snapshot number; also populates 
        cat() instance with relevant attributes.'''
        self.object = il.groupcat.load(self.path, self.snapno)
        self.sf = self.object['header']['Time'] #time (scale factor) of snapshot
        self.hub = self.object['header']['HubbleParam'] #hubble parameter of simulation

        self.X = u.kpc*self.object['halos']['GroupPos'][:,0][:lim]*self.sf/self.hub #spatial coordinates of halos
        self.Y = u.kpc*self.object['halos']['GroupPos'][:,1][:lim]*self.sf/self.hub
        self.Z = u.kpc*self.object['halos']['GroupPos'][:,2][:lim]*self.sf/self.hub

        self.x = u.kpc*self.object['subhalos']['SubhaloPos'][:,0]*self.sf/self.hub #spatial coordinates of subhalos
        self.y = u.kpc*self.object['subhalos']['SubhaloPos'][:,1]*self.sf/self.hub
        self.z = u.kpc*self.object['subhalos']['SubhaloPos'][:,2]*self.sf/self.hub

        if xyzplot:
            '''Plots in 3D the spatial coordinates of [lim] halos, finds all associated subhalos and plots them in 3D.'''
            groupids = np.array(self.object['subhalos']['SubhaloGrNr'])

            assign = []
            for i in np.arange(lim):
                assign.append(np.where(groupids==i)[0])

            subhalono = np.sum([len(assign[i]) for i in range(len(assign))])

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(projection='3d')
            ax.grid(True)
            # ax.scatter(x, y, z)
            ax.scatter(self.X.to('Mpc'), self.Y.to('Mpc'), self.Z.to('Mpc'), marker='o', color='b',s = 10, alpha=0.3, label = 'Halos')

            for i in range(len(assign)):
                ax.scatter(self.x[assign[i]].to('Mpc'), self.y[assign[i]].to('Mpc'), self.z[assign[i]].to('Mpc'), marker='.',s=5, color='r', label = 'Subhalos')
                
            # ax.scatter3D(x, y, z)#, c=sfr, cmap='viridis', s=4)
            # ax.scatter3D(X, Y, Z)
            # ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
            # ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
            # ax.ticklabel_format(axis='z', style='sci',scilimits=(0,0))
            subhalopatch = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='r', markersize=5)
            halopatch = Line2D([0], [0], marker='o', color='k', label='Scatter',markerfacecolor='b', markersize=10)
            ax.set_xlabel(r'x [Mpc]')
            ax.set_ylabel(r'y [Mpc]')
            ax.set_zlabel(r'z [Mpc]')
            ax.legend([subhalopatch, halopatch], [f'{subhalono} Subhalos', f'{lim} Halos'], loc='upper left')
            plt.show()
    
    def subhalo_MST(self, xyzplot=True, masscut=1e8, mode='std'):
        '''Plots the MST of the subhalos in the given object.'''
        uni = 'Mpc'
        stars = (self.object['subhalos']['SubhaloMassType'][:,4]) #stellar mass of subhalos
        mc = masscut*self.hub/1e10 #mass cut for subhalos
        stars_indices = np.where(stars>=mc)[0] #indices of subhalos with stellar mass greater than masscut

        if mode=='std': #standard mode where we plot all subhalos with stellar mass greater than masscut
            x = self.x[stars_indices] #spatial coordinates of subhalos with stellar mass greater than masscut
            y = self.y[stars_indices]
            z = self.z[stars_indices]

        elif mode=='sparse': #sparse mode where we plot 1 in every sampling rate subhalos with stellar mass greater than masscut
            sampling = int(input('Please enter the sampling rate for the sparse mode (Works best between 10 and 100): '))
            x = self.x[stars_indices][::sampling]
            y = self.y[stars_indices][::sampling]
            z = self.z[stars_indices][::sampling]

        elif mode=='sphere': #sphere mode where we plot all subhalos within a sphere of radius r
            r = int(input('Please enter the radius of the sphere in Mpc: '))*u.Mpc
            x = self.x[stars_indices]
            y = self.y[stars_indices]
            z = self.z[stars_indices]
            ctr = 150*u.Mpc #centre of sphere should be the centre of the box
            indices = np.where(np.sqrt((x-ctr)**2+(y-ctr)**2+(z-ctr)**2)<=r)[0]
            x = x[indices]
            y = y[indices]
            z = z[indices]
        
        elif mode=='sampled_sphere': #sampled sphere mode where we plot 1 in every sampling rate subhalos within a sphere of radius r
            sampling = int(input('Please enter the sampling rate for the sampled sphere mode (Works best between 10 and 100): '))
            r = int(input('Please enter the radius of the sphere in Mpc: '))*u.Mpc
            x = self.x[stars_indices]
            y = self.y[stars_indices]
            z = self.z[stars_indices]
            ctr = 150*u.Mpc #centre of sphere should be the centre of the box
            indices = np.where(np.sqrt((x-ctr)**2+(y-ctr)**2+(z-ctr)**2)<=r)[0]
            x = x[indices][::sampling]
            y = y[indices][::sampling]
            z = z[indices][::sampling]

        # Initialise MiSTree MST and plot statistics
        mst = mist.GetMST(x=x.to(uni).value, y=y.to(uni).value, z=z.to(uni).value)
        mst.construct_mst()
        d, l, b, s, l_index, b_index = mst.get_stats(include_index=True)
        
        # begins by binning the data and storing this in a dictionary.
        hmst = mist.HistMST()
        hmst.setup(uselog=True)#num_l_bins=25, num_b_bins=20, num_s_bins=15)
        mst_dict = hmst.get_hist(d, l, b, s)

        # plotting which takes as input the dictionary created before.
        pmst = mist.PlotHistMST()
        pmst.read_mst(mst_dict)
        pmst.plot(usebox=True)

        print(f'Mean Subhalo Separation: {np.round(np.mean(l), 2)} Mpc')

        if mode=='std':
            print(f'Number Density of Subhalos: {np.round(len(x)/(300**3), 5)} Mpc^-3')
        elif mode=='sparse':
            print(f'Number Density of Subhalos: {np.round(len(x)/(300**3*sampling), 5)} Mpc^-3')
        elif mode=='sphere':
            print(f'Number Density of Subhalos: {np.round(len(x)/(4/3*np.pi*r**3), 5)} Mpc^-3')
        elif mode=='sampled_sphere':
            print(f'Number Density of Subhalos: {np.round(len(x)/(4/3*np.pi*r**3*sampling), 5)} Mpc^-3')

        if xyzplot:
            # Plot the MST nodes and edges
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(projection='3d')
            ax.grid(False)
            ax.scatter(x.to('Mpc'), y.to('Mpc'), z.to('Mpc'), marker='.', color='purple',s = 1, alpha=0.1)
            
            for i in range(len(l_index[0])):
                ax.plot([x[l_index[0][i]].to('Mpc').value, x[l_index[1][i]].to('Mpc').value], [y[l_index[0][i]].to('Mpc').value, y[l_index[1][i]].to('Mpc').value], zs = [z[l_index[0][i]].to('Mpc').value, z[l_index[1][i]].to('Mpc').value], color='orange', alpha=0.5)
            
            subhalopatch = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='purple', markersize=5)
            MSTpatch = Line2D([0], [0], marker='o', color='orange', label='Scatter',markerfacecolor='k', markersize=0.1)

            ax.set_title(f'TNG300-1 z=0 Snapshot={self.snapno} Sampling={1/sampling} Radius={r}')
            ax.set_xlabel(r'x [Mpc]')
            ax.set_ylabel(r'y [Mpc]')
            ax.set_zlabel(r'z [Mpc]')
            ax.legend([subhalopatch, MSTpatch], [f'{len(x)} Subhalos', 'Subalo MST'], loc='upper left')
            plt.show()

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
        ax.scatter(X.to('Mpc'), Y.to('Mpc'), Z.to('Mpc'), marker='o', color='b',s = 10, alpha=0.3, label = 'Halos')

        for i in range(len(assign)):
            ax.scatter(x[assign[i]].to('Mpc'), y[assign[i]].to('Mpc'), z[assign[i]].to('Mpc'), marker='.',s=5, color='r', label = 'Subhalos')
            
        # ax.scatter3D(x, y, z)#, c=sfr, cmap='viridis', s=4)
        # ax.scatter3D(X, Y, Z)
        # ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
        # ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        # ax.ticklabel_format(axis='z', style='sci',scilimits=(0,0))
        ax.set_xlabel(r'x [Mpc]')
        ax.set_ylabel(r'y [Mpc]')
        ax.set_zlabel(r'z [Mpc]')
        # fig.legend()
        plt.show()

    return object

def halo_MST(object, lim=500, xyzplot=True):
    '''Plots the MST of the halos in the given object.'''
    sf = object['header']['Time'] #time of snapshot
    hub = object['header']['HubbleParam'] #hubble parameter of simulation
    X = u.kpc*object['halos']['GroupPos'][:,0][:lim]*sf/hub #spatial coordinates of halos
    Y = u.kpc*object['halos']['GroupPos'][:,1][:lim]*sf/hub
    Z = u.kpc*object['halos']['GroupPos'][:,2][:lim]*sf/hub
    groupids = np.array(object['subhalos']['SubhaloGrNr'])

    assign = [] #list of len lim and each element is a list of subhalos associated with that halo
    for i in np.arange(lim):
        assign.append(np.where(groupids==i)[0])

    subhalono = np.sum([len(assign[i]) for i in range(len(assign))])
    print(f'There are {subhalono} subhalos in {lim} halos.')

    x = u.kpc*object['subhalos']['SubhaloPos'][:,0]*sf/hub #spatial coordinates of subhalos
    y = u.kpc*object['subhalos']['SubhaloPos'][:,1]*sf/hub
    z = u.kpc*object['subhalos']['SubhaloPos'][:,2]*sf/hub

    # Initialise MiSTree MST and plot statistics
    mst = mist.GetMST(x=X, y=Y, z=Z)
    mst.construct_mst()
    d, l, b, s, l_index, b_index = mst.get_stats(include_index=True)

    # We want stats on the full dataset
    # X = object['halos']['GroupPos'][:,0][:lim]*sf/hub #spatial coordinates of halos
    # Y = object['halos']['GroupPos'][:,1][:lim]*sf/hub
    # Z = object['halos']['GroupPos'][:,2][:lim]*sf/hub
    uni = 'kpc'
    mst2 = mist.GetMST(x=X.to(uni).value, y=Y.to(uni).value, z=Z.to(uni).value)
    mst2.construct_mst()
    d2, l2, b2, s2, l_index2, b_index2 = mst2.get_stats(include_index=True)
    
    # begins by binning the data and storing this in a dictionary.
    hmst = mist.HistMST()
    hmst.setup(num_l_bins=25, num_b_bins=20, num_s_bins=15)
    mst_dict = hmst.get_hist(d2, l2, b2, s2)

    # plotting which takes as input the dictionary created before.
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(usebox=True)

    if xyzplot:
        # Plot the MST nodes and edges
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        ax.grid(False)
        halos = ax.scatter(X.to('Mpc'), Y.to('Mpc'), Z.to('Mpc'), marker='o', color='b',s = 15, alpha=0.3, label = f'{lim} Halos')        
        
        for i in range(len(l_index[0])):
            ax.plot([X[l_index[0][i]].to('Mpc').value, X[l_index[1][i]].to('Mpc').value], [Y[l_index[0][i]].to('Mpc').value, Y[l_index[1][i]].to('Mpc').value], zs = [Z[l_index[0][i]].to('Mpc').value, Z[l_index[1][i]].to('Mpc').value], color='orange', alpha=0.5)
        
        for i in range(len(assign)):
            ax.scatter(x[assign[i]].to('Mpc'), y[assign[i]].to('Mpc'), z[assign[i]].to('Mpc'), marker='.',s=1, color='r')
        
        subhalopatch = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='r', markersize=5)
        MSTpatch = Line2D([0], [0], marker='o', color='orange', label='Scatter',markerfacecolor='k', markersize=0.1)

        ax.set_xlabel(r'x [Mpc]')
        ax.set_ylabel(r'y [Mpc]')
        ax.set_zlabel(r'z [Mpc]')
        ax.legend([halos, subhalopatch, MSTpatch], [f'{lim} Halos',f'{subhalono} Subhalos', 'Halo MST'], loc='upper left')
        plt.show()
    return mst

def subhalo_MST(object, lim=500000, xyzplot=True):
    '''Plots the MST of the subhalos in the given object.'''
    sf = object['header']['Time'] #time of snapshot
    hub = object['header']['HubbleParam'] #hubble parameter of simulation
    x = u.kpc*object['subhalos']['SubhaloPos'][:,0][:lim]*sf/hub #spatial coordinates of subhalos
    y = u.kpc*object['subhalos']['SubhaloPos'][:,1][:lim]*sf/hub
    z = u.kpc*object['subhalos']['SubhaloPos'][:,2][:lim]*sf/hub
    uni = 'Mpc'

    # test plotting of subhalos
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(projection='3d')
    # ax.grid(False)
    # halos = ax.scatter(x.to('Mpc'), y.to('Mpc'), z.to('Mpc'), marker='o', color='red',s = 15, alpha=0.3, label = f'{lim} Subhalos')
    # ax.set_xlabel(r'x [Mpc]')
    # ax.set_ylabel(r'y [Mpc]')
    # ax.set_zlabel(r'z [Mpc]')
    # plt.show()

    # Initialise MiSTree MST and plot statistics
    mst = mist.GetMST(x=x.to(uni).value, y=y.to(uni).value, z=z.to(uni).value)
    mst.construct_mst()
    d, l, b, s, l_index, b_index = mst.get_stats(include_index=True)
    
    # begins by binning the data and storing this in a dictionary.
    hmst = mist.HistMST()
    hmst.setup(num_l_bins=25, num_b_bins=20, num_s_bins=15)
    mst_dict = hmst.get_hist(d, l, b, s)

    # plotting which takes as input the dictionary created before.
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(usebox=True)

    if xyzplot:
        # Plot the MST nodes and edges
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        ax.grid(False)
        halos = ax.scatter(x.to('Mpc'), y.to('Mpc'), z.to('Mpc'), marker='.', color='red',s = 1, alpha=0.1, label = f'{lim} Subhalos')
        
        for i in range(len(l_index[0])):
            ax.plot([x[l_index[0][i]].to('Mpc').value, x[l_index[1][i]].to('Mpc').value], [y[l_index[0][i]].to('Mpc').value, y[l_index[1][i]].to('Mpc').value], zs = [z[l_index[0][i]].to('Mpc').value, z[l_index[1][i]].to('Mpc').value], color='orange', alpha=0.5)
        
        subhalopatch = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='r', markersize=5)
        MSTpatch = Line2D([0], [0], marker='o', color='orange', label='Scatter',markerfacecolor='k', markersize=0.1)

        ax.set_xlabel(r'x [Mpc]')
        ax.set_ylabel(r'y [Mpc]')
        ax.set_zlabel(r'z [Mpc]')
        ax.legend([subhalopatch, MSTpatch], [f'{lim} Subhalos', 'Subalo MST'], loc='upper left')
        plt.show()

if __name__ == '__main__':
    # test=readsnap(r'/global/homes/d/dkololgi/TNG300-1', 99, xyzplot=False)
    # halo_MST(test, xyzplot=True)
    # subhalo_MST(test, xyzplot=True)

    testcat = cat(path=r'/global/homes/d/dkololgi/TNG300-1', snapno=99)
    testcat.readcat(xyzplot=False)
    testcat.subhalo_MST(xyzplot=True, mode='sparse')