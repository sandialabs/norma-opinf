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



def do_test(settings):
  model_types = ['linear','quadratic','symmetric linear']
  trial_space_splitting_types = ['split','combined']
  acceleration_computation_types = ['finite-difference','acceleration-snapshots']
  boundary_truncation_types = ['size','energy']
  truncation_types = ['size','energy']
  training_skip_steps = [1,2]
  stop_training_times = [0.5,1.0,'end']
  regularization_parameters = ['automatic','list','value'] 
  model_types,trial_space_splitting_types,acceleration_computation_types,boundary_truncation_types , truncation_types , training_skip_steps, stop_training_times, regularization_parameters =  np.meshgrid(model_types,trial_space_splitting_types,acceleration_computation_types,boundary_truncation_types , truncation_types , training_skip_steps, stop_training_times,regularization_parameters)

  n_tests = len(model_types.flatten())
  for i in range(0,n_tests): 
    settings['model-type'] = model_types.flatten()[i]
    settings['trial-space-splitting-type'] = trial_space_splitting_types.flatten()[i]
    settings['stop-training-time'] = stop_training_times.flatten()[i]
    if stop_training_times.flatten()[i] != 'end':
      settings['stop-training-time'] = float(stop_training_times.flatten()[i]) 
    settings['training-skip-steps'] = int(training_skip_steps.flatten()[i])
    settings['forcing'] =  False
    settings['truncation-type'] = truncation_types.flatten()[i]
    settings['boundary-truncation-type'] =  boundary_truncation_types.flatten()[i] 
    settings['trial-space-splitting-type'] = trial_space_splitting_types.flatten()[i]
    settings['acceleration-computation-type'] = acceleration_computation_types.flatten()[i]
    if settings['boundary-truncation-type'] == 'energy':
      settings['boundary-truncation-value'] = 1.e-8
    else:
      settings['boundary-truncation-value'] = 5

    if settings['truncation-type'] == 'energy':
      settings['truncation-value'] = 1.e-8 
    else:
      settings['truncation-value'] = 5

    if regularization_parameters.flatten()[i] == 'automatic':
      settings['regularization-parameter'] = regularization_parameters.flatten()[i] 
    elif regularization_parameters.flatten()[i] == 'list':
      settings['regularization-parameter'] = np.array([1.e-5,1.e-4]) 
    elif regularization_parameters.flatten()[i] == 'value':
      settings['regularization-parameter'] = 5.e-5 

    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    settings['model-name'] = file_path + '/tmp'
    normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict,settings)
    os.system('rm -r ' + file_path + '/tmp')
    os.system('rm ' + file_path + '/tmp.npz')

def test_overlap():
  settings = {}
  settings['fom-yaml-file'] = file_path + "/../data/overlap/cuboid/cuboid-2.yaml"
  settings['training-data-directories'] = [file_path + '/../data/overlap/cuboid/']
  do_test(settings)

def test_overlap_w_multiple_files():
  settings = {}
  settings['fom-yaml-file'] = file_path + "/../data/overlap/cuboid/cuboid-2.yaml"
  settings['training-data-directories'] = [file_path + '/../data/overlap/cuboid/',file_path + '/../data/overlap/cuboid/']
  settings['solution-id'] = 2
  do_test(settings)


def test_single():
  settings = {}
  settings['fom-yaml-file'] = file_path + "/../data/single/cuboid.yaml"
  settings['training-data-directories'] = [file_path + '/../data/single/']
  settings['solution-id'] = 1
  do_test(settings)


#if __name__ == '__main__':
#  test_get_snapshots()
#  #test_get_sidesets()
#  #test_single()
#  #test_overlap()
