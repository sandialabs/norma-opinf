import numpy as np

data = np.load('opinf-operator-gold.npz')
data2  = np.load('opinf-operator.npz')

for key in list(data.keys()):
  assert np.allclose(data[key],data2[key]), print(key,data2[key])
