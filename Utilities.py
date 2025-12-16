import illustris_python as il
import mistree as mist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import astropy.units as u
import astropy.constants as c
import pandas as pd
import networkx as nx
import seaborn as sns
import scienceplots
import sys
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean, minkowski

plt.style.use(['science', 'no-latex'])  # Use dark background for better contrast

# For black background
custom_palette = {
    0: '#80ffdb',  # Void — mint-teal neon (distinct from blue wall)
    1: '#3a86ff',  # Wall — neon blue
    2: '#ff006e',  # Filament — hot pink
    3: '#ffbe0b'   # Cluster — neon yellow-orange
}

class cat():
    def __init__(self, path, snapno=None, masscut=1e9, from_DESI=False):
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
        
        if isinstance(masscut, (int, float)):
            self.masscut = masscut
        else:
            raise TypeError('The masscut must be a number.')
        
        self.from_DESI = from_DESI

        if self.from_DESI:
            self.get_DESI_GAL_CAT()
        else:
            
            if isinstance(snapno, int):
                if len(f'{snapno}') and 0<= snapno <=99:
                    self.snapno = snapno
                else:
                    raise ValueError(f'The snapshot number {snapno} is invalid. Please enter a number from 0 to 99.')
            else:
                raise TypeError('The snapshot number must be an integer.')
            
            self.readcat_obj = self.readcat(xyzplot=False)

    def __repr__(self):
        assert hasattr(self, 'object'), 'No TNG object has been read in. Please add one using the readcat() method.'
        boxsize = round(((u.kpc*self.object['header']['BoxSize']*self.sf/self.hub).to('Mpc')).value)
        redshift = round(self.object['header']['Redshift'])
        return f'TNG{boxsize}Mpc_z={redshift}_Snapshot={self.snapno}'

    def get_DESI_GAL_CAT(self):
        '''Returns the DESI galaxy catalogue object.'''
        sys.path.append("../")

        from galaxy_catalog import GalaxyCatalog

        self.DESI_GAL_CAT = GalaxyCatalog(
            PATH=self.path, # Path to reduced fastspecfit BGS catalog,
            LOGMSTAR=self.masscut # Stellar mass cut for the DESI galaxy catalogue
        )

        # self.DESI_GAL_CAT.cartesian_coord() # obsolete since cartesian coordinates are now generated on initialization
        self.posx = self.DESI_GAL_CAT.X
        self.posxn = self.DESI_GAL_CAT.Xn
        self.posxs = self.DESI_GAL_CAT.Xs

        self.posy = self.DESI_GAL_CAT.Y
        self.posyn = self.DESI_GAL_CAT.Yn
        self.posys = self.DESI_GAL_CAT.Ys

        self.posz = self.DESI_GAL_CAT.Z
        self.poszn = self.DESI_GAL_CAT.Zn
        self.poszs = self.DESI_GAL_CAT.Zs

    def readcat(self, xyzplot, lim=5000):
        '''Reads in groupcat data from the given path and snapshot number; also populates 
        cat() instance with relevant attributes.'''
        self.object = il.groupcat.load(self.path, self.snapno)
        self.sf = self.object['header']['Time'] #time (scale factor) of snapshot
        self.hub = self.object['header']['HubbleParam'] #hubble parameter of simulation

        self.x = u.kpc*self.object['subhalos']['SubhaloPos'][:,0]*self.sf/self.hub #spatial coordinates of subhalos
        self.y = u.kpc*self.object['subhalos']['SubhaloPos'][:,1]*self.sf/self.hub
        self.z = u.kpc*self.object['subhalos']['SubhaloPos'][:,2]*self.sf/self.hub

        # self.x = u.kpc*self.object['subhalos']['SubhaloCM'][:,0]*self.sf/self.hub #spatial coordinates of subhalos
        # self.y = u.kpc*self.object['subhalos']['SubhaloCM'][:,1]*self.sf/self.hub
        # self.z = u.kpc*self.object['subhalos']['SubhaloCM'][:,2]*self.sf/self.hub

        uni = 'Mpc'
        stars = (self.object['subhalos']['SubhaloMassType'][:,4]) #stellar mass of subhalos
        mc = self.masscut*self.hub/1e10 #mass cut for subhalos
        stars_indices = np.where(stars>=mc)[0] #indices of subhalos with stellar mass greater than masscut

        print(f'There are {len(stars_indices)} subhalos with stellar mass greater than {mc}.')

        #standard mode where we plot all subhalos with stellar mass greater than masscut
        x = self.x[stars_indices] #spatial coordinates of subhalos with stellar mass greater than masscut
        y = self.y[stars_indices]
        z = self.z[stars_indices]

        self.posx = x.to(uni).value
        self.posy = y.to(uni).value
        self.posz = z.to(uni).value

        if xyzplot:
            '''Plots in 3D the spatial coordinates of [lim] halos, finds all associated subhalos and plots them in 3D.'''
            groupids = np.array(self.object['subhalos']['SubhaloGrNr'])

            assign = []
            for i in np.arange(lim):
                assign.append(np.where(groupids==i)[0])

            subhalono = np.sum([len(assign[i]) for i in range(len(assign))])

            fig = plt.figure(figsize=(10,10), dpi=300)
            ax = fig.add_subplot(projection='3d')
            ax.grid(False)
            ax.set_facecolor('none')    # makes axes transparent

            # ax.scatter(x, y, z)
            # ax.scatter(self.X.to('Mpc'), self.Y.to('Mpc'), self.Z.to('Mpc'), marker='o', color='b',s = 10, alpha=0.3, label = 'Halos')

            for i in range(len(assign)):
                ax.scatter(self.x[assign[i]].to('Mpc'), self.y[assign[i]].to('Mpc'), self.z[assign[i]].to('Mpc'), marker='.',s=5, color='r', label = 'Subhalos')
                
            # ax.scatter3D(x, y, z)#, c=sfr,s cmap='viridis', s=4)
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

        print(f'There are {len(stars_indices)} subhalos with stellar mass greater than {mc}.')

        if mode=='std': #standard mode where we plot all subhalos with stellar mass greater than masscut
            x = self.x[stars_indices] #spatial coordinates of subhalos with stellar mass greater than masscut
            y = self.y[stars_indices]
            z = self.z[stars_indices]
            title_text = f'TNG300-1 z=0 MST. M/C: $10^{{10}}\,$ M$_{{\odot}}$. Subhalos: {len(x)}'
            # f'TNG300-1 z=0 Snapshot={self.snapno}'

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
        self.posx = x.to(uni).value
        self.posy = y.to(uni).value
        self.posz = z.to(uni).value
        mst = mist.GetMST(x=self.posx, y=self.posy, z=self.posz)
        mst.construct_mst()
        self.d, self.l, self.b, self.s, self.l_index, self.b_index = mst.get_stats(include_index=True)
        # self.tree = mst.tree
        # print(mst.tree)
        # print(type(mst.tree))

        # begins by binning the data and storing this in a dictionary.
        hmst = mist.HistMST()
        hmst.setup(uselog=True)#num_l_bins=25, num_b_bins=20, num_s_bins=15)
        mst_dict = hmst.get_hist(self.d, self.l, self.b, self.s)

        # plotting which takes as input the dictionary created before.
        # pmst = mist.PlotHistMST()
        # pmst.read_mst(mst_dict)
        # pmst.plot(usebox=True)

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
            
            subhalopatch = Line2D([0], [0], marker='.', color='w', label='Scatter',markerfacecolor='purple', markersize=10)
            MSTpatch = Line2D([0], [0], marker='o', color='orange', label='Scatter',markerfacecolor='k', markersize=0.1)

            ax.set_title(title_text)
            ax.set_xlabel(r'x [Mpc]')
            ax.set_ylabel(r'y [Mpc]')
            ax.set_zlabel(r'z [Mpc]')
            ax.legend([subhalopatch, MSTpatch], [f'{len(x)} Subhalos', 'Subalo MST'], loc='upper left')
            ax.set_box_aspect(None, zoom=0.85)
            plt.show()
            fig.savefig('MST_Cube.pdf')


        self.subhalo_table = pd.DataFrame(data = {'x':x.to('Mpc'), 'y':y.to('Mpc'), 'z':z.to('Mpc')})

        self.edge_table = pd.DataFrame(data = {'l':self.l, 'l_start':self.l_index[0], 'l_end':self.l_index[1]})

        # self.adj = np.zeros(shape=(len(self.d), len(self.d)))

        # self.adj[self.edge_table['l_start'].array, self.edge_table['l_end'].array] = 1
        # self.adj[self.edge_table['l_end'].array, self.edge_table['l_start'].array] = 1
    
    def subhalo_complex_network(self, l=2, xyzplot=False):
        '''Produces a network graph of subhalos where all subhalos are connected if their separation is less than the linking length.'''
        # Find the subhalos that are connected
        # self.adj = radius_neighbors_graph(self.subhalo_table[['x', 'y', 'z']], l, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
        self.adj = radius_neighbors_graph(self.subhalo_table[['x', 'y', 'z']], l, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=-1)
        
        # G = nx.from_scipy_sparse_array(self.adj)
        G = nx.from_scipy_sparse_array(self.adj)

        # Add node coordinates as attributes
        for i, coord in enumerate(np.array(self.subhalo_table)):
            G.nodes[i]['pos'] = coord

        # Add edge lengths as attributes
        for u, v in G.edges():
            G.edges[u, v]['length'] = euclidean(np.array(self.subhalo_table)[u], np.array(self.subhalo_table)[v])

        
        if xyzplot:
            '''
            Visualise the subhalos and their connections in 3D.
            '''
            title_text = f'TNG300-1 z=0 Complex (r = ${l}\,Mpc$). M/C: $10^{{10}}\,$ M$_{{\odot}}$. Subhalos: {len(self.subhalo_table)}'

            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(projection='3d')
            ax.grid(False)
            ax.scatter(self.subhalo_table['x'], self.subhalo_table['y'], self.subhalo_table['z'], marker='.', color='purple',s = 1, alpha=0.1)    

            for edge in G.edges():
                ax.plot([self.subhalo_table['x'][edge[0]], self.subhalo_table['x'][edge[1]]], [self.subhalo_table['y'][edge[0]], self.subhalo_table['y'][edge[1]]], zs = [self.subhalo_table['z'][edge[0]], self.subhalo_table['z'][edge[1]]], color='orange', alpha=0.8, linewidth=0.8)
            
            subhalopatch = Line2D([0], [0], marker='.', color='w', label='Scatter',markerfacecolor='purple', markersize=10)
            Complexpatch = Line2D([0], [0], marker='o', color='orange', label='Scatter',markerfacecolor='k', markersize=0.1)
            ax.legend([subhalopatch, Complexpatch], [f'{len(self.subhalo_table)} Subhalos', 'Subalo Complex'], loc='upper left')

            ax.set_title(title_text)
            ax.set_xlabel(r'x [Mpc]')
            ax.set_ylabel(r'y [Mpc]')
            ax.set_zlabel(r'z [Mpc]')
            ax.set_box_aspect(None, zoom=0.85)
            plt.show()
            fig.savefig('Complex_Cube.pdf')

        return G

    def subhalo_delauany_network(self, xyzplot=True, slice = 0.5, tolerance = 10):
        '''Produces a network graph of subhalos using the Delauany triangulation.'''

        if self.from_DESI==False:
            self.points = np.vstack([self.posx, self.posy, self.posz]).T
            self.tri = Delaunay(self.points) # Delauany triangulation of the subhalos
            # self.tri = [(self.points, Delaunay(self.points))] # List of tuples containing the points and the Delauany triangulation object

        elif self.from_DESI==True:
            self.pointsn = np.vstack([self.posxn, self.posyn, self.poszn]).T
            self.pointss = np.vstack([self.posxs, self.posys, self.poszs]).T
            self.points = np.vstack([self.pointsn, self.pointss]) # Combine the north and south galactic hemispheres into one array
            self.trin = Delaunay(self.pointsn) # Delauany triangulation of the subhalos in the north galactic hemisphere
            self.tris = Delaunay(self.pointss) # Delauany triangulation of the subhalos in the south galactic hemisphere
            # self.tri = [(self.pointsn, Delaunay(self.pointsn)), (self.pointss, Delaunay(self.pointss))] # List of tuples containing the points and the Delauany triangulation object for both hemispheres

        if xyzplot & (self.from_DESI==False):
            plt.style.use(['science', 'no-latex'])  # Use dark background for better contrast

            # fig = plt.figure(figsize=(10,10), dpi=300)
            # ax = fig.add_subplot(projection='3d')
            # ax.grid(False)
            # ax.set_facecolor('none')    # makes axes transparent

            # ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c='#ff006e', marker='o', s=1) # Plot the subhalos as points

            # lines = []
            # for simplex in self.tri.simplices: # A simplex is a generalised triangle in n dimensions and tri.simplices is a list of tetrahedra as indices
            #     for i in range(4):
            #         for j in range(i+1, len(simplex)):
            #             lines.append([self.points[simplex[i]], self.points[simplex[j]]])

            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            # subhalopatch = Line2D([0], [0], marker='.', color='white', label='Scatter',markerfacecolor='red', markersize=18)
            # Delpatch = Line2D([0], [0], marker='o', color='#4c78a8', label='Scatter',markerfacecolor='k', markersize=0.1)

            # line_collection = Line3DCollection(lines, colors='#4c78a8', linewidth=0.05, alpha = 0.2)
            # ax.view_init(elev=20, azim=120)
            # ax.add_collection3d(line_collection)
            # ax.set_title(f'TNG300-1 z=0 Delaunay. M/C: $10^{int(np.log10(self.masscut))}\,$ M$_{{\odot}}$. Subhalos: {len(self.points)}', fontsize = 18, pad=10)
            # ax.set_xlabel(r'X [Mpc]', fontsize = 16)
            # ax.set_ylabel(r'Y [Mpc]', fontsize = 16)
            # ax.set_zlabel(r'Z [Mpc]', fontsize = 16)
            # ax.legend([subhalopatch, Delpatch], [f'{len(self.points[:,0])} Subhalos', 'Delaunay'], loc='upper left')
            # ax.set_aspect('equal', adjustable='box')
            # fig.savefig('Delaunay_Cube.pdf')
            # plt.show()

            # Define the z-value for the slice
            z_slice = slice
            # Define a small tolerance for selecting points near the slice

            # Select points near the z_slice
            slice_mask = np.abs(self.points[:, 2] - z_slice) < tolerance
            slice_points = self.points[slice_mask]

            # Filter the simplices to only those within the slice
            slice_lines = []
            for simplex in self.tri.simplices:
                simplex_mask = slice_mask[simplex]
                if np.sum(simplex_mask) >= 2:  # At least two points in the slice
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            if slice_mask[simplex[i]] and slice_mask[simplex[j]]:
                                edge = [self.points[simplex[i]], self.points[simplex[j]]]
                                slice_lines.append(edge)

            # Plotting the 2D slice
            fig, ax = plt.subplots(figsize=(10,10), dpi=600)
            ax.set_facecolor('none')    # makes axes transparent

            # Plot the edges in the slice
            for line in slice_lines:
                ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], '#3a86ff', lw=0.2, zorder=1)

            # Plot the points in the slice
            ax.scatter(slice_points[:, 0], slice_points[:, 1], c='#e17c9a', marker='o', s=13, edgecolors='none', zorder=2)
            # show buffer region of 10 Mpc around box
            # Define the vertices for the outer and inner rectangles
            # Outer: (0,0) -> (300,0) -> (300,300) -> (0,300) -> (0,0) (CCW)
            # Inner: (10,10) -> (290,10) -> (290,290) -> (10,290) -> (10,10) (CW to make a hole)
            vertices = [
                (0, 0), (300, 0), (300, 300), (0, 300), (0, 0), # Outer
                (10, 10), (10, 290), (290, 290), (290, 10), (10, 10) # Inner (CW)
            ]

            codes = [
                Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
                Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY
            ]

            path = Path(vertices, codes)
            patch = PathPatch(path, facecolor='none', hatch='//', edgecolor='#ff006e', alpha=1, zorder=3)
            ax.add_patch(patch)

            ax.set_xlim(0,300)
            ax.set_ylim(0,300)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel(r'X [Mpc]', fontsize = 20)
            ax.set_ylabel(r'Y [Mpc]', fontsize = 20)
            ax.set_title(f'2D Slice at z = {z_slice*300} [Mpc]. M/C: $10^{int(np.log10(self.masscut))}\,$ M$_{{\odot}}$. Subhalos: {len(self.points)}', pad=10, fontsize = 20)
            ax.set_aspect('equal', adjustable='box')
            # ax.legend([subhalopatch, Delpatch], [f'{len(slice_points[:,0])} Subhalos', 'Delaunay'], loc='upper right')
            fig.savefig('Delaunay_Slice.pdf', bbox_inches='tight', dpi=600, transparent=True)
            print('Figure saved as Delaunay_Slice.pdf')
            plt.show()

        if self.from_DESI == False:
            # Create a networkx graph from the Delaunay triangulation
            G = nx.Graph()

            # Add the nodes
            for i, point in enumerate(self.points):
                G.add_node(i, pos=point)
                G.nodes[i]['pos'] = point

            # Calculating the edge lengths from points using scipy's euclidean distance
            for i in range(len(self.tri.simplices)): # For each simplex
                for j in range(4): # For each point in the simplex
                    for k in range(j+1, 4): # For each other point in the simplex
                        G.add_edge(self.tri.simplices[i][j], self.tri.simplices[i][k], length=euclidean(self.points[self.tri.simplices[i][j]], self.points[self.tri.simplices[i][k]]))

            return G
        
        elif self.from_DESI == True:
            # Create a networkx graph from the Delaunay triangulation
            G = nx.Graph()

            # Add the nodes for the north galactic hemisphere
            for i, point in enumerate(self.pointsn):
                G.add_node(i, pos=point)
                G.nodes[i]['pos'] = point
            for i, point in enumerate(self.pointss):
                G.add_node(i + len(self.pointsn), pos=point)
                G.nodes[i + len(self.pointsn)]['pos'] = point
          
            # Calculating the edge lengths from points using scipy's euclidean distance
            for i in range(len(self.trin.simplices)):
                for j in range(4):
                    for k in range(j+1, 4):
                        G.add_edge(self.trin.simplices[i][j], self.trin.simplices[i][k], length=euclidean(self.pointsn[self.trin.simplices[i][j]], self.pointsn[self.trin.simplices[i][k]]))
            for i in range(len(self.tris.simplices)):
                for j in range(4):
                    for k in range(j+1, 4):
                        idx_j = self.tris.simplices[i][j] + len(self.pointsn)
                        idx_k = self.tris.simplices[i][k] + len(self.pointsn)
                        G.add_edge(idx_j, idx_k, length=euclidean(self.pointss[self.tris.simplices[i][j]], self.pointss[self.tris.simplices[i][k]]))

            return G
        
    def edge_classification(self, x, y, z):
        '''Classifies the edges of the MST of the subhalos in the given object.'''
        # Classify the edges of the MST according to MiSTree
        # self.MST_x_edges = np.array([(x[self.l_index[0]]/self.dx).astype(int), (x[self.l_index[1]]/self.dx).astype(int)])
        # self.MST_y_edges = np.array([(y[self.l_index[0]]/self.dx).astype(int), (y[self.l_index[1]]/self.dx).astype(int)])
        # self.MST_z_edges = np.array([(z[self.l_index[0]]/self.dx).astype(int), (z[self.l_index[1]]/self.dx).astype(int)])

        # classifications = self.cwebdata[self.MST_x_edges, self.MST_y_edges, self.MST_z_edges] # Classifications of the MST edges
        classifications = (self.cweb[self.l_index[0]], self.cweb[self.l_index[1]])
        self.start = classifications[0] # classifications of the start of the edges
        self.end = classifications[1]
        self.notcross_boundary = np.where(self.start == self.end)[0] # Edges that do not cross a cosmic web classification boundary
        self.cross_boundary = np.where(self.start != self.end)[0] # Edges that do cross a cosmic web classification boundary

        # Need to find how much of an edge is in one boundary or another
        # Find the midpoint of cross_boundary edge
        self.midpoints = np.array([(x[self.l_index[0][self.cross_boundary]]+x[self.l_index[1][self.cross_boundary]])/2, (y[self.l_index[0][self.cross_boundary]]+y[self.l_index[1][self.cross_boundary]])/2, (z[self.l_index[0][self.cross_boundary]]+z[self.l_index[1][self.cross_boundary]])/2])
        
        # Classification of midpoints
        self.mid_classifications = self.cwebdata[(self.midpoints[0]/self.dx).astype(int), (self.midpoints[1]/self.dx).astype(int), (self.midpoints[2]/self.dx).astype(int)]
        
        # Classification of midpoints same as start or end?
        # If the classification of the midpoints is the same as the start then the edge must have the same
        # CWEB classification as the start, and vice versa
        mask = self.mid_classifications == self.start[self.cross_boundary]

        self.classifications = classifications[0]
        self.classifications[self.cross_boundary] = classifications[0][self.cross_boundary]*mask + classifications[1][self.cross_boundary]*(~mask) # Correcting the classificaitons of the boudnary crossing edges. ~ is the bitwise NOT operator

        # Plot the MST edges with their cosmic web classifications
        #edges = np.delete(self.l, self.cross_boundary)
        void_edges = self.l[list(set(np.where(self.start == 0)[0]))]
        wall_edges = self.l[list(set(np.where(self.start == 1)[0]))]
        filament_edges = self.l[list(set(np.where(self.start == 2)[0]))]
        cluster_edges = self.l[list(set(np.where(self.start == 3)[0]))]

        fig = plt.figure(figsize=(16,8))
        ax = plt.subplot()
        bins = 50 # 100
        ax.hist(void_edges, bins=bins, alpha=0.25, density = True, color='r')
        sns.kdeplot(data=void_edges, alpha=1, label=f'Void ({len(void_edges)})', color='r')
        ax.hist(wall_edges, bins=bins, alpha=0.25, density = True, color='g')
        sns.kdeplot(data=wall_edges, alpha=1, label=f'Wall ({len(wall_edges)})', color='g')
        ax.hist(filament_edges, bins=bins, alpha=0.25, density = True, color='b')
        sns.kdeplot(data=filament_edges, alpha=1, label=f'Filament ({len(filament_edges)})', color='b')
        ax.hist(cluster_edges, bins=bins, alpha=0.25, density = True, color='y')
        sns.kdeplot(data=cluster_edges, alpha=1, label=f'Cluster ({len(cluster_edges)})', color='y')
        ax.legend(prop={'size':20})
        ax.set_xlabel(r'Edge length [$Mpc$]')
        ax.set_ylabel('Frequency')
        ax.set_title(f'MST Edge Length Distributions. {self.filetype} Cosmic Web. Subhalos: {len(x)}')
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        fig.savefig('MST_Edge_Length_Distributions.pdf')
        plt.show()

    def degree_classification(self):
        '''Classifies the degree of the subhalos in the given object.'''
        # Classify the degree of the subhalos according to MiSTree
        void_d = self.d[np.where(self.classifications == 0)[0]]
        wall_d = self.d[np.where(self.classifications == 1)[0]]
        filament_d = self.d[np.where(self.classifications == 2)[0]]
        cluster_d = self.d[np.where(self.classifications == 3)[0]]

        # Plot the degree distributions
        fig = plt.figure(figsize=(16,8))
        ax = plt.subplot()
        ax.hist(void_d, alpha=0.5, density = False, label=f'Void ({len(void_d)})')
        ax.hist(wall_d, alpha=0.5, density = False, label=f'Wall ({len(wall_d)})')
        ax.hist(filament_d, alpha=0.5, density = False, label=f'Filament ({len(filament_d)})')
        ax.hist(cluster_d, alpha=0.5, density = False, label=f'Cluster ({len(cluster_d)})')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel(r'Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distributions')
        plt.show()

    def branch_classification(self):
        '''Classifies the branch length of the subhalos in the given object.
        
        This method classifies the branch length of the subhalos based on the given object's MST (Minimum Spanning Tree).
        It prints the number of branches in the MST and assigns a classification value to each subhalo based on the
        classifications dictionary.
        '''
        print(f"There are {len(self.b)} branches in the MST.")
        # Classify the branch length of the subhalos according to MiSTree
        # for i in range(len(self.b_index)):
        #     for j in range(len(self.b_index[i])):
        #         if self.classifications[self.b_index[i][j]] == 0:
        #             self.b_index[i][j] = 0
        #         elif self.classifications[self.b_index[i][j]] == 1:
        #             self.b_index[i][j] = 1
        #         elif self.classifications[self.b_index[i][j]] == 2:
        #             self.b_index[i][j] = 2
        #         elif self.classifications[self.b_index[i][j]] == 3:
        #             self.b_index[i][j] = 3

        # Do a weighted average of the branch lengths based on the classifications of the constituent edges

        self.branch_edge_classification = [self.classifications[self.b_index[i]] for i in range(len(self.b_index))]
        self.branch_edge_lengths = [self.l[self.b_index[i]] for i in range(len(self.b_index))]
        self.branch_edge_length_ratios = [self.branch_edge_lengths[i]/np.sum(self.branch_edge_lengths[i]) for i in range(len(self.branch_edge_lengths))]
        self.summy = np.array(self.branch_edge_length_ratios,dtype=object)*np.array(self.branch_edge_classification,dtype=object)
        self.branch_classification = np.array([np.round(np.sum(self.summy[i])) for i in range(len(self.summy))],dtype=int)
        
        # Plot the branch length distributions
        void_b = self.b[np.where(self.branch_classification == 0)[0]]
        wall_b = self.b[np.where(self.branch_classification == 1)[0]]
        filament_b = self.b[np.where(self.branch_classification == 2)[0]]
        cluster_b = self.b[np.where(self.branch_classification == 3)[0]]
        fig = plt.figure(figsize=(16,8))
        ax = plt.subplot()
        ax.hist(void_b, alpha=0.5, bins=100, density = False, label=f'Void ({len(void_b)})')
        ax.hist(wall_b, alpha=0.5, bins=100, density = False, label=f'Wall ({len(wall_b)})')
        ax.hist(filament_b, alpha=0.5, bins=100, density = False, label=f'Filament ({len(filament_b)})')
        ax.hist(cluster_b, alpha=0.5, bins=100, density = False, label=f'Cluster ({len(cluster_b)})')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel(r'Branch Length')
        ax.set_ylabel('Frequency')
        ax.set_title('Branch Length Distributions [$Mpc$]')
        plt.show()


    def cweb_classify(self, xyzplot=True):
        '''Plots the cosmic web classications of the subhalos in the given object.'''
        self.cwebfile = np.load(r'/global/homes/d/dkololgi/TNG/Illustris/TNG300_snap_099_tweb_env_merged.npz') #TNG300_snap_099_tweb_env_merged.npz
        # self.significances = np.load(r'C:\Users\dkter\OneDrive - University College London\Year 1\MST\new_TNG300_snap_099_nexus_sig_merged.npz')        
        self.filetype = 'T-Web' # Nexus+
        self.cwebdata = self.cwebfile['cweb']
        # self.Sc = self.significances['Sc']
        # self.Sf = self.significances['Sf']
        # self.Sw = self.significances['Sw']
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
        # self.Sc_subhalos = self.Sc[self.xpix, self.ypix, self.zpix]
        # self.Sf_subhalos = self.Sf[self.xpix, self.ypix, self.zpix]
        # self.Sw_subhalos = self.Sw[self.xpix, self.ypix, self.zpix]

        if xyzplot:
            self.reds = np.count_nonzero(self.cweb == 0) # Voids
            self.greens = np.count_nonzero(self.cweb == 1) # Walls
            self.blues = np.count_nonzero(self.cweb == 2) # Filaments
            self.yellows = np.count_nonzero(self.cweb == 3) # Clusters

            colors = np.array([custom_palette[c] for c in self.cweb])

            fig = plt.figure(figsize=(10,10), dpi=300)
            ax = fig.add_subplot(projection='3d')
            ax.grid(False)
            ax.set_facecolor('none')
            
            # Add a plane at z=150 Mpc
            x_min, x_max = 0, 300  # Box limits
            y_min, y_max = 0, 300
            z_plane = 150
            
            xx, yy = np.meshgrid(np.array([x_min, x_max]), np.array([y_min, y_max]))
            zz = np.full_like(xx, z_plane)
            
            ax.plot_surface(xx, yy, zz, alpha=1., color='#1f2b24')#, color='#080080')
            
            ax.scatter(x.to('Mpc'), y.to('Mpc'), z.to('Mpc'), marker='.', color=list(colors), s = 1, alpha=0.5)
            ax.set_xlim(0, 300)
            ax.set_ylim(0, 300)
            ax.set_zlim(0, 300)
            ax.set_xlabel(r'X [Mpc]', fontsize=16, labelpad=10)
            ax.set_ylabel(r'Y [Mpc]', fontsize=16, labelpad=10)
            ax.set_zlabel(r'Z [Mpc]', fontsize=16, labelpad=10)
            ax.tick_params(axis='both', labelsize=16)
            mc = np.log10(self.masscut)
            ax.set_title(f'TNG300-1 z=0 {self.filetype}. M/C [log$_{{10}}$(M$_{{\odot}})]$: ${(int(np.log10(self.masscut)))}\,$. Subhalos: {len(x)}', fontsize=18)
            subhalopatch_void = Line2D([0], [0], marker='.', color='none', label='Scatter',markerfacecolor=custom_palette[0], markersize=10)
            subhalopatch_wall = Line2D([0], [0], marker='.', color='none', label='Scatter',markerfacecolor=custom_palette[1], markersize=10)
            subhalopatch_filament = Line2D([0], [0], marker='.', color='none', label='Scatter',markerfacecolor=custom_palette[2], markersize=10)
            subhalopatch_cluster = Line2D([0], [0], marker='.', color='none', label='Scatter',markerfacecolor=custom_palette[3], markersize=10)
            ax.set_box_aspect(None, zoom=0.85)
            ax.set_aspect('auto', adjustable='box')
            fig.savefig(f'TNG300-1_z=0_{self.filetype}_Subhalos.png', transparent=True)
            # print(f'Figure saved as TNG300-1_z=0_{self.filetype}_Subhalos.png')
            plt.show()

            plt.rcdefaults()
            plt.style.use(['science', 'no-latex', 'dark_background'])  # Use dark background for better contrast with custom colors

            # 2D projection plot of DESI galaxies with cosmic web predictions
            zlims = (-10, 10)*u.Mpc  # Set z slab limits in Mpc
            fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)
            
            ax = fig.add_subplot()
            # set z slab between -10 and 10 Mpc
            mask = (z.to('Mpc') >= zlims[0]) & (z.to('Mpc') <= zlims[1])
            ax.scatter(x[mask].to('Mpc'), y[mask].to('Mpc'), c=[custom_palette[c] for c in self.cweb[mask]], s=5, edgecolor='none')
            ax.set_facecolor('none')    # makes axes transparent

            # Set axis labels and title
            # ax.legend(handles=[
            # plt.Line2D([0], [0], marker='o', color='w', label=environ_dicts[i],
            #    markerfacecolor=custom_palette[i], markersize=5) for i in range(4)
            # ], loc='best')

            ax.set_xlim(0, 300)
            ax.set_ylim(0, 300)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('X [Mpc]', fontsize=16, labelpad=10)
            ax.set_ylabel('Y [Mpc]', fontsize=16, labelpad=10)
            ax.set_title('IllustrisTNG-300 by T-WEB Environments', fontsize=18, pad=10)
            # Set aspect ratio to equal for better visualization
            ax.set_aspect('equal', adjustable='box')
            fig.savefig(f'TNG300-1_z=0_{self.filetype}_Subhalos_2D_Projection.png', transparent=True)
            # Show the plot
            plt.show()
        
        # self.edge_classification(x=x, y=y, z=z)
        # self.degree_classification()
        # self.branch_classification()

    def cross_plots(self, x = 'branch length', y = 'edge length'):
        if x == 'branch length' and y == 'edge length':
            '''
            Plot of branch length vs edge length, coloured by classification
            '''
            fig = plt.figure(figsize=(16,8))
            ax = plt.subplot()
            for i in range(len(self.branch_classification)):
                if self.branch_classification[i] == 0:
                    ax.scatter(np.repeat(self.b[i], len(self.branch_edge_lengths[i])), self.branch_edge_lengths[i], marker='.', color='r')
                elif self.branch_classification[i] == 1:
                    ax.scatter(np.repeat(self.b[i], len(self.branch_edge_lengths[i])), self.branch_edge_lengths[i], marker='.', color='g')
                # elif self.branch_classification[i] == 2:
                #     ax.scatter(np.repeat(self.b[i], len(self.branch_edge_lengths[i])), self.branch_edge_lengths[i], marker='.', color='b')
                elif self.branch_classification[i] == 3:
                    ax.scatter(np.repeat(self.b[i], len(self.branch_edge_lengths[i])), self.branch_edge_lengths[i], marker='.', color='y')
            # Plot a x = y line for reference
            ax.plot(np.linspace(0, 10, 100), np.linspace(0, 10, 100), '--', color='white')
            ax.set_ylabel(r'Edge Length [$Mpc$]')
            ax.set_xlabel(r'Branch Length [$Mpc$]')
            subhalopatchr = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='red', markersize=20)
            subhalopatchg = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='green', markersize=20)
            subhalopatchb = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='blue', markersize=20)
            subhalopatchy = Line2D([0], [0], marker='.', color='k', label='Scatter',markerfacecolor='yellow', markersize=20)
            ax.legend([subhalopatchr, subhalopatchg, subhalopatchb, subhalopatchy], [f'Void ({np.count_nonzero(self.cweb == 0)})', f'Wall ({np.count_nonzero(self.cweb == 1)})', f'Filamentary ({np.count_nonzero(self.cweb == 2)})', f'Cluster ({np.count_nonzero(self.cweb == 3)})'], loc='upper left')
            plt.show()

        # if x == 'degree' and y == 'edge length':
        #     '''
        #     Plot of degree vs edge length, coloured by classification
        #     '''

    def networkx(self):
        
        
        return nx.from_scipy_sparse_array(self.tree)

    def visualise_netx(self):
        G = self.networkx()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.show()
    # def MST_degree(self):
    #     degree_dist = np.sum(self.adj, axis=0)
    #     return degree_dist
    
    # def MST_

        
    # def cw_sig(self):
    #     self.sigs = np.load(r'/Users/daksheshkololgi/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 1/MST/new_TNG300_snap_099_nexus_sig_merged.npz')





if __name__ == '__main__':
    # test=readsnap(r'/global/homes/d/dkololgi/TNG300-1', 99, xyzplot=False)
    # halo_MST(test, xyzplot=True)
    # subhalo_MST(test, xyzplot=True)
    testcat = cat(path=r'/pscratch/sd/d/dkololgi/TNG300-1', snapno=99, masscut=1e9)#'/global/homes/d/dkololgi/TNG300-1'
    # self.readcat_obj(xyzplot=False)
    # testcat.subhalo_MST(xyzplot=True, mode='std')
    # cweb = testcat.cweb_classify()
    # testcat.cross_plots()
    testcat.subhalo_delauany_network(xyzplot=True)