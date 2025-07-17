import numpy as np
import opinf
import os
from matplotlib import pyplot as plt
import normaopinf
import normaopinf.readers
import normaopinf.calculus
import normaopinf.parser
import normaopinf.opinf
import romtools
import scipy.optimize
import argparse
import pathlib
import pytest
file_path = str(pathlib.Path(__file__).parent.resolve())

def formatRed(skk):
  return "\033[91m {}\033[00m" .format(skk)


def test_get_sidesets():
  fom_yaml = normaopinf.parser.open_yaml(file_path + "/../data/overlap/cuboid/cuboid-2.yaml")
  sidesets = normaopinf.opinf.get_bc_sidesets(fom_yaml)
  truth_sidesets = ['nsx--x','nsy--y','nsz+-z','ssz-'] 
  assert(sidesets == truth_sidesets)

def test_get_snapshots():
  settings = {}
  settings['fom-yaml-file'] = file_path + "/../data/overlap/cuboid/cuboid-2.yaml"
  settings['training-data-directories'] = [file_path + '/../data/overlap/cuboid/']
  settings['training-skip-steps'] = 1
  settings['stop-training-time'] = 'end' 

  ## Check assertion errors
  with pytest.raises(AssertionError, match="fom-yaml-file must be a string"):
      settings['fom-yaml-file'] = 5
      normaopinf.opinf.get_processed_snapshots(settings) 
  settings['fom-yaml-file'] = file_path + "/../data/overlap/cuboid/cuboid-2.yaml"

  with pytest.raises(AssertionError, match="training-data-directories must be a list"):
      settings['training-data-directories'] = 'test'
      normaopinf.opinf.get_processed_snapshots(settings) 
  settings['training-data-directories'] = [file_path + '/../data/overlap/cuboid/']

  snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)


