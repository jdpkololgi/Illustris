import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import h5py
import numpy as np
from astropy.cosmology import Planck18

from workflows.sbi import build_p12a_matched_response as writer
from workflows.abacus_tweb.p3br_build_random_response import galactic_cap


class ResponseWriterTests(unittest.TestCase):
    def test_small_full_writer_and_idempotent_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);phase_root=root/'ph001';phase_root.mkdir()
            def save(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            rows=np.array([(20.,-30.,.1),(20.,-30.,.10001)],dtype=[('RA','f8'),('DEC','f8'),('Z','f8')])
            catalogue=root/'observed.fits';fitsio.write(catalogue,rows)
            radius=Planck18.comoving_distance(rows['Z']).value
            ra=np.deg2rad(rows['RA']);dec=np.deg2rad(rows['DEC'])
            xyz=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))*radius[:,None]
            cap=int(galactic_cap(rows['RA'],rows['DEC'])[0]);name=('SGC','NGC')[cap]
            points_path=root/'points.npy';np.save(points_path,np.column_stack((xyz,np.full(len(rows),cap))))
            completion=root/'CATALOGUE_COMPLETE.json'
            save(completion,dict(**{'pass':True},phase='ph001',geometry=dict(units='Mpc',cosmology='Astropy Planck18'),
                artifacts=dict(points=str(points_path),points_sha256=writer.sha256(points_path))))
            p1=phase_root/'p1_canonical/manifest.json'
            save(p1,dict(parent=str(catalogue),parent_sha256=writer.sha256(catalogue),points=str(points_path),
                         source_completion_marker=str(completion),source_completion_marker_sha256=writer.sha256(completion)))
            angular=root/'angular.npz';n=12*256**2
            np.savez(angular,support=np.ones(n,bool),domain=np.full(n,2*cap,'u1'),angular_response=np.ones(n,'f4'))
            field=root/'raw.h5';shape=(4,4,4);cell=5.;origin=xyz[0]-10.
            with h5py.File(field,'w') as handle:
                for key in ('exposure_binary','exposure_apodized','los_x','los_y','los_z'):
                    handle[key]=np.ones(shape,'f4')
            fields=phase_root/'p3_fields/field_manifest.json'
            save(fields,dict(p1_manifest_sha256=writer.sha256(p1),angular_support=dict(source=str(angular),sha256=writer.sha256(angular)),
                components={name:dict(file=str(field),file_sha256=writer.sha256(field),grid=dict(shape=list(shape),origin_mpc=origin.tolist(),cell_mpc=cell))}))
            save(phase_root/'RAW_PREPARATION_COMPLETE.json',dict(pass_raw_preparation=True,fields_sha256=writer.sha256(fields)))
            policy=dict(training=['ph000','ph001'],development_and_calibration=['ph006'])
            selection=root/'point_selection_candidates_v1';selection.mkdir()
            stats={}
            for phase in policy['training']:
                path=selection/(phase+'.json');save(path,dict(phase=phase));stats[phase]=writer.sha256(path)
            fit_hashes={}
            for omitted in (None,'ph000','ph001'):
                fit_name='full' if omitted is None else 'oof_'+omitted
                phases=[p for p in policy['training'] if p!=omitted];path=selection/(fit_name+'.json')
                value=dict(fit_phases=phases,omitted_phase=omitted,density_unit='Mpc^-3',
                    bin_edges=np.linspace(0.,.6,121).tolist(),density=np.full((2,120),.001).tolist(),
                    valid=np.ones((2,120),bool).tolist(),statistics_sha256={p:stats[p] for p in phases})
                save(path,value);fit_hashes[fit_name]=writer.sha256(path)
            save(selection/'CANDIDATES_COMPLETE.json',dict(pipeline='p12',policy=policy,fits=fit_hashes))
            with patch.object(writer,'require_compute'),patch.object(writer.p12,'load_plan',return_value=(dict(output_root=str(root)),{},policy)),\
                    patch.object(writer.p12,'require_matching_domains'),patch.dict(writer.os.environ,SLURM_JOB_ID='test'):
                writer.build('ph001',[cap],1)
                marker=phase_root/'matched_cic_response_v1'/(name+'_COMPLETE.json')
                record=json.loads(marker.read_text());original=marker.read_bytes()
                with h5py.File(record['output']['path'],'r') as handle:
                    self.assertAlmostEqual(float(handle['counts'][:].sum()),2.,places=5)
                    weights=np.stack([handle['basis_'+str(i)][:] for i in range(3)])
                    expected=writer.basis.expected_counts(weights,handle['radial_bin'][:],value['density'][cap],value['valid'][cap])
                    np.testing.assert_allclose(expected,.125,rtol=2e-6)
                self.assertFalse(record['model_inputs_ready'])
                writer.build('ph001',[cap],1)
                self.assertEqual(marker.read_bytes(),original)


if __name__=='__main__':unittest.main()
