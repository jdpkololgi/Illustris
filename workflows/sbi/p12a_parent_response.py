"""Exact P12-A random-support features at the registered P3 voxel centres.

Only support and boundary distance enter the P12-A posterior. Evaluate those
same grid functions at the voxels addressed by parent rows, avoiding unrelated
response channels and a full 3-D overlay. No galaxy labels enter this operation.
"""
from pathlib import Path
import json
import numpy as np
import healpy as hp
from workflows.abacus_tweb import p3br_build_random_response as b
from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256


def sample_centres(points, component, cap_support, boundary_angle, selection):
    spec = b.grid_from_component(component)
    index = np.floor((np.asarray(points[:,:3],dtype=np.float64)-spec.origin)/spec.cell_mpc).astype(np.int64)
    inside = np.all((index>=0)&(index<np.asarray(spec.shape)),axis=1)
    distance = np.full(len(points),np.nan,dtype=np.float32)
    support = np.zeros(len(points),dtype=bool)
    selected=np.flatnonzero(inside)
    xyz=np.asarray(spec.origin)+(index[selected].astype(np.float64)+.5)*spec.cell_mpc
    radius=np.sqrt(xyz[:,0]*xyz[:,0]+xyz[:,1]*xyz[:,1]+xyz[:,2]*xyz[:,2])
    safe=np.maximum(radius,1e-12)
    pix=hp.vec2pix(hp.npix2nside(len(cap_support)),xyz[:,0]/safe,xyz[:,1]/safe,xyz[:,2]/safe,nest=False)
    redshift=b.redshift_from_radius(radius,selection)
    binary=(redshift>=.10)&(redshift<.60)&~((redshift>=.585)&(redshift<.595))&cap_support[pix]
    angular=boundary_angle[pix].astype(np.float64)*radius
    boundaries=np.interp([.10,.60,.585,.595],selection['cosmology']['redshift_grid'],selection['cosmology']['radius_grid_mpc'])
    radial=np.minimum(np.minimum(abs(radius-boundaries[0]),abs(radius-boundaries[1])),
                      np.minimum(abs(radius-boundaries[2]),abs(radius-boundaries[3])))
    distance[selected]=np.minimum(angular,radial).astype(np.float32)*binary
    support[selected]=binary
    return distance,support


def prepare_parent_response(root, contract, phase, angular_path, output):
    root,contract,angular_path,output=map(Path,(root,contract,angular_path,output))
    p3_path=root/phase/'p3_fields/field_manifest.json'; p3=json.loads(p3_path.read_text())
    p1_path=root/phase/'p1_canonical/manifest.json'
    selection_path=contract/'transforms/field/selection_manifest.json'
    selection=json.loads(selection_path.read_text())
    angular=b.load_map(angular_path)
    if sha256(angular_path) != json.loads(angular_path.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Angular payload checksum mismatch')
    points_path=Path(p3['points']); points=np.load(points_path,mmap_mode='r')
    output.mkdir(parents=True,exist_ok=True)
    paths={'distance':output/'distance_by_parent.npy','support':output/'support_by_parent.npy'}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Parent response arrays already exist without reusable terminal approval')
    distance=np.lib.format.open_memmap(paths['distance'],mode='w+',dtype=np.float32,shape=(len(points),))
    support=np.lib.format.open_memmap(paths['support'],mode='w+',dtype=bool,shape=(len(points),))
    distance[:]=np.nan; support[:]=False
    caps=np.asarray(points[:,3],dtype=np.uint8)
    if not np.isin(caps,[0,1]).all():raise ValueError('Unknown cap ID')
    for cap,name in b.CAPS:
        cap_support=np.asarray(angular['support'],dtype=bool)&((angular['domain']//2)==cap)
        angle=b.angular_boundary_distance(cap_support)
        rows=np.flatnonzero(caps==cap)
        for start in range(0,len(rows),250000):
            chosen=rows[start:start+250000]
            distance[chosen],support[chosen]=sample_centres(points[chosen],p3['components'][name],cap_support,angle,selection)
    distance.flush();support.flush()
    supported=np.asarray(support)
    if not supported.any() or not np.isfinite(distance[supported]).all() or (distance[supported]<0).any():
        raise ValueError('Invalid supported parent response values')
    record={'schema_version':'p12a-parent-response-v1','phase':phase,'pass':True,'rows':len(points),
            'support_rows':int(supported.sum()),'representation':'parent-indexed exact P3 voxel lookup',
            'p1_manifest_sha256':sha256(p1_path),'p3_manifest_sha256':sha256(p3_path),
            'points':str(points_path),'points_sha256':sha256(points_path),
            'angular_map':str(angular_path),'angular_map_sha256':sha256(angular_path),
            'selection_sha256':sha256(selection_path),'source_sha256':sha256(Path(__file__)),
            'arrays':{key:str(path) for key,path in paths.items()},
            'array_sha256':{key:sha256(path) for key,path in paths.items()},'labels_read':False}
    path=output/'P12A_PARENT_RESPONSE_READY.json';atomic_json(path,record)
    return path,record
