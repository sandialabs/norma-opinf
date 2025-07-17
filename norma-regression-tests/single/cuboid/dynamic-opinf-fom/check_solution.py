import numpy as np

gold = np.genfromtxt('cuboid-disp-0020.gold',delimiter=',')
data = np.genfromtxt('cuboid-disp-0020.csv',delimiter=',')

assert np.allclose(data,gold)
