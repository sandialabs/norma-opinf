import numpy as np
import sys
def d2dx2(f : np.ndarray, x: np.ndarray, method='2nd-order-uniform-dx'):
  '''
  Compute the second derivative of f evaluate on the grid x

  Args:
      f (np.ndarray): samples of the function of shape (...,nx)
      x (np.ndarray): 1d array containing values of x at which f is sampled

  Returns:
      fxx: second derivative of f w.r.p. to x
  '''

  if method == '2nd-order-uniform-dx':  
      assert np.allclose( np.linalg.norm(np.diff(np.diff(x))) , 0. ), "Only a uniform time grid is supported for this method"
      dx = x[1,0] - x[0,0]
      fpp = np.zeros(f.shape)
      fpp[...,1:-1] = (f[...,2::] - 2.*f[...,1:-1] + f[...,0:-2]) / dx**2
      fpp[...,0] = (2.*f[...,0] - 5.*f[...,1] + 4.*f[...,2] - f[...,3] ) / dx**2 
      fpp[...,-1] = (2.*f[...,-1] - 5.*f[...,-2] + 4.*f[...,-3] - f[...,-4] ) / dx**2
      return fpp
  else:
     print("Method " + method + " not supported")
     sys.exit() 


if __name__ == '__main__':
    x = np.linspace(0,5,50)
    f = 5*x**2 + 7*x + 3
    print(x)
    print(f)
    fpp = np.ones(x.size)*10.
    fpp_numerical = d2dx2(f[None],x)
    assert np.allclose(fpp,fpp_numerical)
