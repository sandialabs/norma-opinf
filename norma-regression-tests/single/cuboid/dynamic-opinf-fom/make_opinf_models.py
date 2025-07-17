import normaopinf
import normaopinf.opinf
import os
import numpy as np

if __name__ == '__main__':
    settings = {}
    settings['fom-yaml-file'] = 'cuboid.yaml'
    settings['training-data-directories'] = [os.getcwd()]
    settings['model-type'] = 'linear'
    settings['stop-training-time'] = 100000.0
    settings['training-skip-steps'] = 1
    settings['forcing'] = False
    settings['truncation-type'] = 'energy'
    settings['boundary-truncation-type'] = 'energy'
    settings['regularization-parameter'] = 'automatic' 
    settings['model-name'] = 'opinf-operator'
    settings['truncation-value'] = 0.999999
    settings['boundary-truncation-value'] = 0.999999
    settings['trial-space-splitting-type'] = 'split'
    settings['acceleration-computation-type'] = 'finite-difference'
    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
