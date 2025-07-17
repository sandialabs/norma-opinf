import normaopinf
import normaopinf.opinf
import nnopinf
import numpy as np

if __name__ == '__main__':
  settings = {}
  # Link to FOM yaml file, which tells us about sidesets
  settings['fom-yaml-file'] = "../run_0/torsion.yaml"

  # List of training data directories
  td = []
  for i in range(0,10):
    td.append('../run_' + str(i) + '/')
  settings['training-data-directories'] = td

  # ID of norma domains
  settings['solution-id'] = 1

  # Specify model type (linear, quadratic, linear-symmetric, neural network)
  settings['model-type'] = 'linear'

  # When to stop training. If you want to use all steps, put "end" or a high number
  settings['stop-training-time'] = 3.e-3

  # Setps to skip in training
  settings['training-skip-steps'] = 10

  # If we want to learn a "forcing" operator (e.g., OpInf-cA instead of OpInf-A)
  settings['forcing'] =  False

  # How to do trunctation (size/energy)
  settings['truncation-type'] = 'size'
  settings['boundary-truncation-type'] =  'energy'

  # Regularization parameter (float, or "automatic" to do grid search)
  settings['regularization-parameter'] =  5.e-3

  # Option to split the trial space (split/combined)
  settings['trial-space-splitting-type'] = 'split'

  # How to compute acceleration (finite-difference or acceleration-snapshots)
  settings['acceleration-computation-type'] = 'acceleration-snapshots'

  # Load in snapshots
  snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)

  # In this example, loop over the ROM dimension of the OpInf model
  sizes = np.array([5,10,15],dtype=int)
  for i,size in enumerate(sizes):
    settings['truncation-value'] = int(size)
    settings['boundary-truncation-value'] = 1. - 1.e-5
    # Where the model gets saved to
    settings['model-name'] = 'linear-model-' + str(size)
    normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict,settings)
~                                                                                  
