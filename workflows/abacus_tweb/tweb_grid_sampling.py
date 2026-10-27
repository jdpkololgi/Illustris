"""Bounded memory-mapped T-Web sampling, including periodic derivative stencils."""
from pathlib import Path
import struct
import zipfile
import numpy as np

LABEL_COLUMNS=('LAMBDA1','LAMBDA2','LAMBDA3',
 'DLAM1_DX','DLAM1_DY','DLAM1_DZ','DLAM2_DX','DLAM2_DY','DLAM2_DZ',
 'DLAM3_DX','DLAM3_DY','DLAM3_DZ','LAP_LAM1','LAP_LAM2','LAP_LAM3')

def mapped_member(path,key):
    """Map stored numeric NPY members; reject compressed/object data explicitly."""
    with zipfile.ZipFile(path) as z:
        info=z.getinfo(key+'.npy')
        if info.compress_type!=zipfile.ZIP_STORED or info.flag_bits&1:
            raise ValueError('T-Web numeric members must be stored uncompressed and unencrypted')
        with z.open(info) as f:
            version=np.lib.format.read_magic(f)
            reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
            if reader is None: raise ValueError('Unsupported NPY header')
            shape,fortran,dtype=reader(f); header_bytes=f.tell()
        if dtype.hasobject or header_bytes+int(np.prod(shape))*dtype.itemsize!=info.file_size:
            raise ValueError('Invalid numeric array size/dtype')
        with Path(path).open('rb') as f:
            f.seek(info.header_offset); raw=f.read(30)
        if raw[:4]!=b'PK\x03\x04': raise ValueError('Invalid local ZIP header')
        name_bytes,extra_bytes=struct.unpack_from('<HH',raw,26)
        offset=info.header_offset+30+name_bytes+extra_bytes+header_bytes
    return np.memmap(path,mode='r',dtype=dtype,shape=shape,offset=offset,order='F' if fortran else 'C')

class TWebGrid:
    def __init__(self,directory):
        self.directory=Path(directory).resolve(); self.slabs=[]; self.cache={}
        definitions=set()
        for p in sorted(self.directory.glob('abacus_cactus_tweb_rank*.npz')):
            with np.load(p,allow_pickle=False) as a:
                definition=(int(a['ngrid']),float(a['boxsize']),float(a['threshold']),float(a['Rsmooth']))
                definitions.add(definition)
                self.slabs.append((int(a['x_start']),int(a['x_end']),p))
        if len(definitions)!=1: raise ValueError('Missing/inconsistent T-Web grid definitions')
        self.ngrid,self.boxsize,self.threshold,self.rsmooth=definitions.pop()
        self.slabs.sort()
        expected=0
        for lo,hi,p in self.slabs:
            if lo!=expected or hi<=lo or hi>self.ngrid: raise ValueError('T-Web slab gap/overlap')
            expected=hi
        if expected!=self.ngrid: raise ValueError('Incomplete T-Web slab coverage')
    def _arrays(self,lo,hi,p):
        if p not in self.cache:
            eig=mapped_member(p,'eig_vals'); cls=mapped_member(p,'cweb')
            if eig.shape!=(3,hi-lo,self.ngrid,self.ngrid) or cls.shape!=(hi-lo,self.ngrid,self.ngrid):
                raise ValueError(f'T-Web member shape mismatch: {p}')
            self.cache[p]=(eig,cls)
        return self.cache[p]
    def _sample(self,indices,classes=False):
        q=np.asarray(indices,dtype=np.int64)%self.ngrid
        eig_out=np.full((len(q),3),np.nan,dtype=np.float32)
        cls_out=np.full(len(q),255,dtype=np.uint8) if classes else None
        for lo,hi,p in self.slabs:
            m=(q[:,0]>=lo)&(q[:,0]<hi)
            if not m.any(): continue
            eig,cls=self._arrays(lo,hi,p)
            x,y,z=q[m].T
            eig_out[m]=eig[:,x-lo,y,z].T
            if classes: cls_out[m]=cls[x-lo,y,z]
        if not np.isfinite(eig_out).all(): raise ValueError('Nonfinite T-Web samples')
        return eig_out,cls_out
    def sample_indices(self,indices):
        q=np.asarray(indices)
        if q.ndim!=2 or q.shape[1]!=3 or not np.issubdtype(q.dtype,np.integer):
            raise ValueError('Grid indices must be integer N by 3')
        center,cls=self._sample(q,classes=True)
        if np.any(np.diff(center,axis=1)<0) or np.any(cls>3):
            raise ValueError('Unordered eigenvalues or invalid CWEB')
        result={name:center[:,i] for i,name in enumerate(LABEL_COLUMNS[:3])}
        result['CWEB']=cls
        lap=np.zeros_like(center)
        h=self.boxsize/self.ngrid
        for axis,letter in enumerate('XYZ'):
            direction=np.zeros(3,dtype=np.int64);direction[axis]=1
            plus,_=self._sample(q+direction);minus,_=self._sample(q-direction)
            grad=(plus-minus)*(self.rsmooth/(2*h))
            lap+=plus+minus-2*center
            for i in range(3): result[f'DLAM{i+1}_D{letter}']=grad[:,i]
        lap*=self.rsmooth**2/h**2
        for i in range(3): result[f'LAP_LAM{i+1}']=lap[:,i]
        return result
    def definition(self):
        return dict(path=str(self.directory),ngrid=self.ngrid,boxsize=self.boxsize,
                    threshold=self.threshold,rsmooth=self.rsmooth,slabs=len(self.slabs),
                    units='Mpc/h',sampling='nearest_cell',derivatives='periodic_central_differences_scaled_by_R_or_R_squared')
