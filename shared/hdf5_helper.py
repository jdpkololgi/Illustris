import numpy as np
import h5py

def print_hdf5_item_structure(item, offset='    '):
    """Prints the input file/group/dataset (item) name recursively.

    Parameters
    ----------
    item : str
        File structure item.
    """
    if isinstance(item, h5py.File):
        print(item.file, '(File)', item.name)
    elif isinstance(item, h5py.Dataset):
        print('(Dataset)', item.name, '    len =', item.shape)
    elif isinstance(item, h5py.Group):
        print('(Group)', item.name)
    else:
        print('Unknown item in HDF5 file:', item.name)
    if isinstance(item, h5py.File) or isinstance(item, h5py.Group):
        for key, val in dict(item).items():
            sub_item = val
            print(offset, key)
            print_hdf5_item_structure(sub_item, offset + '    ')


def get_hdf5_keys(hdf5_filename, overide_extension=False):
    """Prints the HDF5 file structure and keys.

    Parameters
    ----------
    hdf5_filename : str
        Filename of the hdf5 file.
    overide_extension : bool
        Checks extension is hdf5 and allow for this to be added if not included in
        the filename.
    """
    if hdf5_filename.endswith('.hdf5') != True and overide_extension == True:
        hdf5_filename = hdf5_filename + '.hdf5'
    hdf5_file = h5py.File(hdf5_filename, 'r')
    print_hdf5_item_structure(hdf5_file)
    hdf5_file.close()


def get_hdf5_data(hdf5_filename, key_name, overide_extension=False):
    """Outputs a specific data set from the hdf5 data set.

    Parameters
    ----------
    hdf5_filename : str
        Filename of the hdf5 file.
    key_name : str
        the key_name or key_names of items in the hdf5 file that are to be outputted.
    overide_extension : bool
        Checks extension is hdf5 and allow for this to be added if not included in
        the filename.

    Returns
    -------
    data : array
        If key_name is a string then the associated array from the hdf5 file is outputted,
        if a list then a list of arrays is given.
    """
    islist = False
    if isinstance(key_name, list) is True:
        islist = True
    if hdf5_filename.endswith('.hdf5') != True and overide_extension == True:
        hdf5_filename = hdf5_filename + '.hdf5'
    hdf5_file = h5py.File(hdf5_filename, 'r')
    if islist is False:
        data = np.array(hdf5_file[key_name])
    else:
        data = []
        for i in range(0, len(key_name)):
            data.append(np.array(hdf5_file[key_name[i]]))
    return data

if __name__ == '__main__':
    get_hdf5_keys('IllustrisTNG_135_01.hdf5')
    data = get_hdf5_data('IllustrisTNG_135_01.hdf5', 'Coordinates')
    print(data.shape)