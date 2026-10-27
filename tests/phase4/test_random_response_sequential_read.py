import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import fitsio
import numpy as np


class SequentialRandomReadTests(unittest.TestCase):
    def test_contiguous_fits_read_matches_selected_row_column_read(self):
        names=['RA','DEC','PHOTSYS','GOODHARDLOC','MASKBITS']
        values=np.zeros(101,dtype=[('RA','f8'),('DEC','f8'),('PHOTSYS','S1'),('GOODHARDLOC','?'),('MASKBITS','i8'),('UNUSED','f4')])
        values['RA']=np.linspace(0,359,101);values['DEC']=np.linspace(-89,89,101)
        values['PHOTSYS']=np.where(np.arange(101)%2,b'N',b'S')
        values['GOODHARDLOC']=np.arange(101)%3!=0;values['MASKBITS']=np.arange(101)%4
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'random.fits';fitsio.write(path,values,clobber=True)
            with fitsio.FITS(path) as handle:
                for start,stop in [(0,17),(17,61),(61,101)]:
                    expected=handle[1].read(columns=names,rows=np.arange(start,stop,dtype=np.int64))
                    actual=handle[1].read_slice(start,stop)
                    for name in names:np.testing.assert_array_equal(actual[name],expected[name])

if __name__=='__main__':unittest.main()
