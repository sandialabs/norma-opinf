import numpy as np
gold = np.genfromtxt('cuboid-2-reduced_states-0020.gold')
data = np.genfromtxt('cuboid-2-reduced_states-0020.csv')

assert np.allclose(data,gold), print(data)
