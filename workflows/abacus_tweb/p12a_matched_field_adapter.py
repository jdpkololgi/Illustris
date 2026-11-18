"""P12 U-Net patch view over frozen source-matched CIC response bases."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from workflows.abacus_tweb.p6_field_patch_utils import CanonicalFieldPatchAdapter,CAP_NAME,SELECTION_CHANNELS
from workflows.sbi.p12a_cic_response_basis import expected_counts


SELECTION_SCHEMA='p12a-matched-cic-selection-v1'
DEPENDENCIES=('counts','exposure_binary','basis_0','basis_1','basis_2','radial_bin')


def derive_channels(raw,selection,cap,epsilon=.001):
    density=np.asarray(selection['density'][cap],float)
    valid=np.asarray(selection['valid'][cap],bool)
    floating=raw['radial_bin'];bins=np.asarray(floating,dtype=np.int64)
    if not np.array_equal(bins,floating):raise ValueError('Nonintegral radial-bin map')
    weights=np.stack([raw['basis_'+str(i)] for i in range(3)])
    expected=expected_counts(weights,bins,density,valid)
    selected=(raw['exposure_binary']>0)&valid[bins]
    contrast=np.zeros_like(expected)
    if epsilon<=0 or not np.isfinite(epsilon) or np.any(raw['counts']<0):
        raise ValueError('Invalid count/contrast definition')
    contrast[selected]=np.log((raw['counts'][selected]+epsilon)/(expected[selected]+epsilon))
    return dict(expected_counts=expected.astype('f4'),log_count_ratio=contrast.astype('f4'),
                ntilde_mpc3=np.where(selected,density[bins],0.).astype('f4'))


class MatchedFieldPatchAdapter(CanonicalFieldPatchAdapter):
    def __init__(self,root,*,selection_manifest,rotation=0):
        super().__init__(root)
        self.matched_manifest=json.loads(Path(selection_manifest).read_text())
        m=self.matched_manifest
        if (m.get('schema')!=SELECTION_SCHEMA or rotation!=0 or not m.get('pass')
                or m.get('density_unit')!='Mpc^-3' or m.get('cosmology')!='Planck18'
                or m.get('fit_phases')!=m.get('selection',{}).get('fit_phases')
                or m.get('omitted_phase')!=m.get('selection',{}).get('omitted_phase')):
            raise ValueError('Unqualified matched-CIC selection manifest')
        phase=self.manifest.get('phase')
        record=m['phase_response'].get(phase)
        if record is None:raise ValueError('Phase not bound by matched selection manifest')
        for name in CAP_NAME.values():
            declared=self.manifest['caps'][name]
            if (declared['field_path']!=record[name]['path']
                    or declared['field_sha256']!=record[name]['sha256']):
                raise ValueError('Matched response and field adapter differ')

    def extract_bounds(self,**kwargs):
        names=tuple(kwargs.get('channel_names') or self.channel_names)
        load=tuple(dict.fromkeys([name for name in names if name not in SELECTION_CHANNELS]+list(DEPENDENCIES)))
        raw=super().extract_bounds(**dict(kwargs,channel_names=load))
        values=dict(zip(raw.channel_names,raw.values))
        values.update(derive_channels(values,self.matched_manifest['selection'],int(kwargs['cap']),
                                      self.matched_manifest['contrast_epsilon']))
        result=np.stack([values[name] for name in names]).astype('f4')
        if not np.isfinite(result).all():raise ValueError('Nonfinite matched P12 patch')
        return replace(raw,channel_names=names,values=result)
