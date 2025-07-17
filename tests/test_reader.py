import numpy as np
import normaopinf
import normaopinf.readers

import pathlib
file_path = str(pathlib.Path(__file__).parent.resolve())

def test_load_overlap_displacement_csv_files():
    csv_files,times = normaopinf.readers.load_displacement_csv_files(file_path + '/data/overlap/cuboid/','cuboid-1',skip_files=1)
    assert np.allclose(times[-1],1.0)
    assert np.allclose(csv_files[...,-1,-1],np.array([-0.11924440996081176,-0.11924440996081276,0.0]) )
    assert csv_files.shape[-1] == 21

    csv_files_skip,times_skip = normaopinf.readers.load_displacement_csv_files(file_path + '/data/overlap/cuboid/','cuboid-1',skip_files=2)
    assert np.allclose(csv_files[...,::2],csv_files_skip)
    assert np.allclose(times[::2],times_skip)


def test_load_sideset_displacement_csv_files():
    sidesets = ['nsx--x','nsy--y','nsz+-z','ssz-']
    sideset_csv_files = normaopinf.readers.load_sideset_displacement_csv_files(file_path + '/data/overlap/cuboid/',sidesets,'cuboid-2',skip_files=1)
    for sideset in sidesets:
      assert( sideset in sideset_csv_files.keys() )
       
    assert( np.allclose(sideset_csv_files['nsx--x'][...,-1], np.zeros(9))) 

def test_load_displacement_csv_files():
    csv_files,times = normaopinf.readers.load_displacement_csv_files(file_path + '/data/single/','cuboid',skip_files=1)
    assert np.allclose(times[-1],1.0)
    assert np.allclose(csv_files[...,-1,-1],np.array([-0.1668159268781887,-0.16681592687818886,0.0]) )
    assert csv_files.shape[-1] == 21

    csv_files_skip,times_skip = normaopinf.readers.load_displacement_csv_files(file_path + '/data/single/','cuboid',skip_files=2)
    assert np.allclose(csv_files[...,::2],csv_files_skip)
    assert np.allclose(times[::2],times_skip)

def test_load_velocity_csv_files():
    csv_files,times = normaopinf.readers.load_velocity_csv_files(file_path + '/data/single/','cuboid',skip_files=1)
    assert np.allclose(times[-1],1.0)
    assert csv_files.shape[-1] == 21
    assert np.allclose(csv_files[...,-1,-1],np.array([0.009359007477109937,0.009359007478148662,0.0]) ) , csv_files[...,-1,-1]

    csv_files_skip,times_skip = normaopinf.readers.load_velocity_csv_files(file_path + '/data/single/','cuboid',skip_files=2)
    assert np.allclose(csv_files[...,::2],csv_files_skip)
    assert np.allclose(times[::2],times_skip)


def test_load_acceleration_csv_files():
    csv_files,times = normaopinf.readers.load_acceleration_csv_files(file_path + '/data/single/','cuboid',skip_files=1)
    assert np.allclose(times[-1],1.0)
    assert np.allclose(csv_files[...,-1,-1],np.array([20.239909896422326,20.239909914598897,0.0]) )
    assert csv_files.shape[-1] == 21
    csv_files_skip,times_skip = normaopinf.readers.load_acceleration_csv_files(file_path + '/data/single/','cuboid',skip_files=2)
    assert np.allclose(csv_files[...,::2],csv_files_skip)
    assert np.allclose(times[::2],times_skip)

if __name__=='__main__':
    test_load_displacement_csv_files() 
    test_load_overlap_displacement_csv_files() 
    test_load_velocity_csv_files() 
    test_load_acceleration_csv_files() 
    test_load_sideset_displacement_csv_files() 

