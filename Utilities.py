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
    def __init__(self, path, snapno, masscut=1e8):
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
        
        if isinstance(masscut, (int, float)):
            self.masscut = masscut
        else:
            raise TypeError('The masscut must be a number.')
        
        self.readcat = self.readcat(xyzplot=False)
        self.subhalo_MST = self.subhalo_MST(xyzplot=False, mode='std')
        
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
    
    def subhalo_MST(self, xyzplot=True, mode='std'):
        '''Plots the MST of the subhalos in the given object.'''
        uni = 'Mpc'
        stars = (self.object['subhalos']['SubhaloMassType'][:,4]) #stellar mass of subhalos
        mc = self.masscut*self.hub/1e10 #mass cut for subhalos
        stars_indices = np.where(stars>=mc)[0] #indices of subhalos with stellar mass greater than masscut

        if mode=='std': #standard mode where we plot all subhalos with stellar mass greater than masscut
            x = self.x[stars_indices] #spatial coordinates of subhalos with stellar mass greater than masscut
            y = self.y[stars_indices]
            z = self.z[stars_indices]
            title_text = f'TNG300-1 z=0 Snapshot={self.snapno}'

        elif mode=='sparse': #sparse mode where we plot 1 in every sampling rate subhalos with stellar mass greater than masscut
            sampling = int(input('Please enter the sampling rate for the sparse mode (Works best between 10 and 100): '))
            x = self.x[stars_indices][::sampling]
            y = self.y[stars_indices][::sampling]
            z = self.z[stars_indices][::sampling]
            title_text = f'TNG300-1 z=0 Snapshot={self.snapno} Sampling={1/sampling}'

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
            title_text = f'TNG300-1 z=0 Snapshot={self.snapno} Radius={r}'
        
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
            title_text = f'TNG300-1 z=0 Snapshot={self.snapno} Sampling={1/sampling} Radius={r}'

        # Initialise MiSTree MST and plot statistics
        mst = mist.GetMST(x=x.to(uni).value, y=y.to(uni).value, z=z.to(uni).value)
        mst.construct_mst()
        self.d, self.l, self.b, self.s, self.l_index, self.b_index = mst.get_stats(include_index=True)
        
        # begins by binning the data and storing this in a dictionary.
        hmst = mist.HistMST()
        hmst.setup(uselog=True)#num_l_bins=25, num_b_bins=20, num_s_bins=15)
        mst_dict = hmst.get_hist(self.d, self.l, self.b, self.s)

        # plotting which takes as input the dictionary created before.
        pmst = mist.PlotHistMST()
        pmst.read_mst(mst_dict)
        pmst.plot(usebox=True)

        print(f'Mean Subhalo Separation: {np.round(np.mean(self.l), 2)} Mpc')

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
            
            for i in range(len(self.l_index[0])):
                ax.plot([x[self.l_index[0][i]].to('Mpc').value, x[self.l_index[1][i]].to('Mpc').value], [y[self.l_index[0][i]].to('Mpc').value, y[self.l_index[1][i]].to('Mpc').value], zs = [z[self.l_index[0][i]].to('Mpc').value, z[self.l_index[1][i]].to('Mpc').value], color='orange', alpha=0.5)
            
            subhalopatch = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='purple', markersize=5)
            MSTpatch = Line2D([0], [0], marker='o', color='orange', label='Scatter',markerfacecolor='k', markersize=0.1)

            ax.set_title(title_text)
            ax.set_xlabel(r'x [Mpc]')
            ax.set_ylabel(r'y [Mpc]')
            ax.set_zlabel(r'z [Mpc]')
            ax.legend([subhalopatch, MSTpatch], [f'{len(x)} Subhalos', 'Subalo MST'], loc='upper left')
            plt.show()

    def cweb(self, xyzplot=True):
        '''Plots the cosmic web classications of the subhalos in the given object.'''
        self.cwebfile = np.load('/global/homes/d/dkololgi/TNG/Illustris/TNG300_snap_099_nexus_env_merged.npz')
        self.cwebdata = self.cwebfile['cweb']
        ngrid = self.cwebdata.shape[0]
        self.boxsize = u.kpc*self.object['header']['BoxSize']*self.sf/self.hub
        self.dx = self.boxsize/ngrid

        # Create a grid of points
        stars = (self.object['subhalos']['SubhaloMassType'][:,4]) #stellar mass of subhalos
        mc = self.masscut*self.hub/1e10 #mass cut for subhalos
        stars_indices = np.where(stars>=mc)[0] #indices of subhalos with stellar mass greater than masscut

        # Get the spatial coordinates of the subhalos above the masscut
        x = self.x[stars_indices]
        y = self.y[stars_indices]
        z = self.z[stars_indices]        

        # Convert to cweb coordinates
        self.xpix = (x/self.dx).astype(int)
        self.ypix = (y/self.dx).astype(int)
        self.zpix = (z/self.dx).astype(int)

        # Get the cweb classifications of the subhalos
        self.cweb = self.cwebdata[self.xpix, self.ypix, self.zpix]
        colors = np.empty(len(self.cweb), dtype=str)
        reds = np.count_nonzero(self.cweb == 0) # Voids
        greens = np.count_nonzero(self.cweb == 1) # Walls
        blues = np.count_nonzero(self.cweb == 2) # Filaments
        yellows = np.count_nonzero(self.cweb == 3) # Clusters

        colors[self.cweb == 0] = 'r'
        colors[self.cweb == 1] = 'g'
        colors[self.cweb == 2] = 'b'
        colors[self.cweb == 3] = 'y'

        # Plot the cosmic web classifications
        if xyzplot:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(projection='3d')
            ax.grid(False)
            ax.scatter(x.to('Mpc'), y.to('Mpc'), z.to('Mpc'), marker='.', color=colors, s = 1, alpha=0.5)
            ax.set_xlabel(r'x [Mpc]')
            ax.set_ylabel(r'y [Mpc]')
            ax.set_zlabel(r'z [Mpc]')
            ax.set_title(f'TNG300-1 z=0 Snapshot={self.snapno} {len(x)} Subhalos Cosmic Web')
            subhalopatchr = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='red', markersize=10)
            subhalopatchg = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='green', markersize=10)
            subhalopatchb = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='blue', markersize=10)
            subhalopatchy = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='yellow', markersize=10)
            ax.legend([subhalopatchr, subhalopatchg, subhalopatchb, subhalopatchy], [f'{reds} Void', f'{greens} Wall', f'{blues} Filamentary', f'{yellows} Cluster'], loc='upper left')
            plt.show()
               
        # Classify the edges of the MST according to MiSTree
        self.MST_x_edges = np.array([(x[self.l_index[0]]/self.dx).astype(int), (x[self.l_index[1]]/self.dx).astype(int)])
        self.MST_y_edges = np.array([(y[self.l_index[0]]/self.dx).astype(int), (y[self.l_index[1]]/self.dx).astype(int)])
        self.MST_z_edges = np.array([(z[self.l_index[0]]/self.dx).astype(int), (z[self.l_index[1]]/self.dx).astype(int)])

        classifications = self.cwebdata[self.MST_x_edges, self.MST_y_edges, self.MST_z_edges] # Classifications of the MST edges
        start = classifications[0]
        end = classifications[1]
        self.notcross_boundary = np.where(start == end)[0] # Edges that do not cross a cosmic web classification boundary
        self.cross_boundary = np.where(start != end)[0] # Edges that do cross a cosmic web classification boundary

        # Need to find how much of an edge is in one boundary or another using pythagoras

        # Plot the edge statistics
        #edges = np.delete(self.l, self.cross_boundary)
        void_edges = self.l[list(set(np.where(start == 0)[0]))]
        wall_edges = self.l[list(set(np.where(start == 1)[0]))]
        filament_edges = self.l[list(set(np.where(start == 2)[0]))]
        cluster_edges = self.l[list(set(np.where(start == 3)[0]))]

        fig = plt.figure(figsize=(16,8))
        ax = plt.subplot()
        # ax.hist(void_edges, bins=50, alpha=0.5, density = True, label=f'Void ({len(void_edges)})')
        ax.hist(wall_edges, bins=100, alpha=0.5, density = True, label=f'Wall ({len(wall_edges)})')
        ax.hist(filament_edges, bins=100, alpha=0.5, density = True, label=f'Filament ({len(filament_edges)})')
        ax.hist(cluster_edges, bins=100, alpha=0.5, density = True, label=f'Cluster ({len(cluster_edges)})')
        ax.legend()
        ax.set_xlabel(r'Edge length [$Mpc$]')
        ax.set_ylabel('Frequency')
        ax.set_title('MST Edge Length Distributions')
        plt.show()
        



if __name__ == '__main__':
    # test=readsnap(r'/global/homes/d/dkololgi/TNG300-1', 99, xyzplot=False)
    # halo_MST(test, xyzplot=True)
    # subhalo_MST(test, xyzplot=True)

    testcat = cat(path=r'/global/homes/d/dkololgi/TNG300-1', snapno=99, masscut=1e10)
    # testcat.readcat(xyzplot=False)
    # testcat.subhalo_MST(xyzplot=True, mode='std')
    cweb = testcat.cweb()