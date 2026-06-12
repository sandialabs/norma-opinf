# import os

import normaopinf
import normaopinf.opinf

# import numpy as np

if __name__ == "__main__":

    settings = {}
    settings["fom-yaml-file"] = (
        "/home/andiaz/projects/nosam-opinf/Norma_NO-SAM_for_AlejandroD/tension-specimen/fom-fom-0.005/with-output/dynamic-linear-elastic-fom-fom-hex-hex-conformal/impl-impl-dt-1em2-DN-predictor/relax0_1/tension-specimen-2.yaml"
    )
    settings["training-data-directories"] = [
        "/home/andiaz/projects/nosam-opinf/Norma_NO-SAM_for_AlejandroD/tension-specimen/fom-fom-0.005/with-output/dynamic-linear-elastic-fom-fom-hex-hex-conformal/impl-impl-dt-1em2-DN-predictor/relax0_1/"
    ]
    settings["model-type"] = "linear"
    settings["stop-training-time"] = "end"
    settings["training-skip-steps"] = 1
    settings["forcing"] = False

    settings["input-scale"] = "rms"
    # settings["input-scale"] = "none"

    settings["truncation-type"] = "size"
    settings["truncation-value"] = 12
    settings["boundary-truncation-type"] = "size"
    settings["boundary-truncation-value"] = 5

    # settings["truncation-type"] = "energy"
    # settings["truncation-value"] = 1 - 1e-12
    # settings["boundary-truncation-type"] = "energy"
    # settings["boundary-truncation-value"] = 1 - 1e-12

    # settings['regularization-parameter'] =  1.0e-4
    settings["regularization-parameter"] = [
        1.0e-10,
        1.0e-9,
        1.0e-8,
        1.0e-7,
        1.0e-6,
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0e0,
        1.0e1,
        1.0e2,
    ]
    # settings['trial-space-splitting-type'] = 'split'
    settings["trial-space-splitting-type"] = "combined"

    # settings['acceleration-computation-type'] = 'finite-difference'
    settings["acceleration-computation-type"] = "acceleration-snapshots"

    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    settings["model-name"] = "linear-opinf-2"
    normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
