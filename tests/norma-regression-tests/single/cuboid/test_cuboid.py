import os
import pytest
import numpy as np
import subprocess

# Run the command

def test_cuboid(request):
  norma = request.config.getoption("--norma")
  current_directory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(current_directory + '/dynamic-opinf-fom')
  #subprocess.run([norma, 'cuboid.yaml'], check=True)
  norma_list = norma.split(' ')
  norma_list.append("cuboid.yaml")
  subprocess.run(norma_list)
  subprocess.run(['python', 'make_opinf_models.py'], check=True)
  data = np.load('opinf-operator.npz')
  data2 = np.load('opinf-operator-gold.npz')
  # Use np.abs to account for potential sign flipping
  for key in list(data.keys()):
    if key.split('-')[-1] == 'cutoff':
      pass
    else: 
      assert(np.allclose(np.abs(data[key]),np.abs(data2[key])))
  os.chdir('../dynamic-opinf-rom')
  subprocess.run(['cp','../dynamic-opinf-fom/opinf-operator.npz','.'])
  subprocess.run(norma_list)
  gold = np.genfromtxt('cuboid-reduced_states-0020.gold')
  data = np.genfromtxt('cuboid-reduced_states-0020.csv')
  assert np.allclose(np.abs(gold),np.abs(data))

if __name__=='__main__':
  test_cuboid()

  
