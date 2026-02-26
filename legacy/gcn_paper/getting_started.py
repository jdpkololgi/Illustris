import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import astropy.units as u
import astropy.constants as c
import scienceplots
plt.style.use('science')

#Test data
basePath = './Illustris-3/output/'
fields = ['SubhaloMass','SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath,135,fields=fields)
print(subhalos.keys())
print(subhalos['SubhaloMass'].shape)

# mass_msun = subhalos['SubhaloMass'] * 1e10 / 0.704
# plt.plot(mass_msun,subhalos['SubhaloSFRinRad'],'.')
# plt.xscale('log')
# plt.yscale('log')
# plt.yscale('log')
# plt.xlabel('Total Mass [$M_\odot$]')
# plt.ylabel('Star Formation Rate [$M_\odot / yr$]')
# plt.show()

GroupFirstSub = il.groupcat.loadHalos(basePath,135,fields=['GroupFirstSub'])
ptNumGas = il.snapshot.partTypeNum('gas') # 0
ptNumStars = il.snapshot.partTypeNum('stars') # 4
for i in range(5):
    all_fields = il.groupcat.loadSingle(basePath,135,subhaloID=GroupFirstSub[i])
    gas_mass   = all_fields['SubhaloMassInHalfRadType'][ptNumGas]
    stars_mass = all_fields['SubhaloMassInHalfRadType'][ptNumStars]
    frac = gas_mass / (gas_mass + stars_mass)
    print(GroupFirstSub[i], frac)

#Illustris300simulation data
basePath2 = r'C:\Users\dkter\OneDrive - University College London\Year 1\Illustris\TNG300-1'
object = il.groupcat.load(basePath2,99)
print(object.keys())

# plt.plot(object['subhalos']['SubhaloMass'], object['subhalos']['SubhaloSFRinRad'],'x')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

sf = object['header']['Time'] #time (scale factor) of snapshot
hub = object['header']['HubbleParam'] #hubble parameter of simulation

#3D plot
# lim = 5000

# sfr = object['subhalos']['SubhaloSFRinRad'][:lim]
masscut = 1e9 #mass cut for subhalos
stars = (object['subhalos']['SubhaloMassType'][:,4]) #stellar mass of subhalos
mc = masscut*hub/1e10 #mass cut for subhalos
stars_indices = np.where(stars>=mc)[0] #indices of subhalos with stellar mass greater than masscut
x = (u.kpc*object['subhalos']['SubhaloPos'][:,0]*sf/hub)[stars_indices]
y = (u.kpc*object['subhalos']['SubhaloPos'][:,1]*sf/hub)[stars_indices]
z = (u.kpc*object['subhalos']['SubhaloPos'][:,2]*sf/hub)[stars_indices]

df = {'x':x, 'y':y, 'z':z}

# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=4, color=sfr, opacity=0.8))])
# fig.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x.to('Mpc'), y.to('Mpc'), z.to('Mpc'), marker='.',s=5, color='r', label = 'Subhalos')




# ax.scatter3D(x, y, z, c=sfr, cmap='viridis', s=4)
# ax.scatter3D(x, y, z, s=4)
# fig.colorbar(ax.scatter3D(x, y, z, c=np.log(sfr), cmap='viridis', s=4), ax=ax)
plt.show()

# fig2 = px.scatter_3d(df, x='x', y='y', z='z', color='sfr', opacity=0.8, size_max=2)
# fig2.show()