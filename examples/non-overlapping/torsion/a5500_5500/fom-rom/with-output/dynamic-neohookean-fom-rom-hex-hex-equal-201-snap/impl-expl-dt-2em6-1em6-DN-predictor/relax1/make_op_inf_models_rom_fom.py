# import os
import normaopinf
import normaopinf.opinf
import numpy as np

# import numpy as np

if __name__ == "__main__":

    settings = {}
    settings["fom-yaml-file"] = (
        "../../../../../fom-fom/with-output/dynamic-neohookean-fom-fom-hex-hex-equal-201-snap/impl-expl-dt-2em6-1em6-DN-predictor/relax1/torsion-2.yaml"
    )
    settings["training-data-directories"] = [
        "../../../../../fom-fom/with-output/dynamic-neohookean-fom-fom-hex-hex-equal-201-snap/impl-expl-dt-2em6-1em6-DN-predictor/relax1/"
    ]
    settings["model-type"] = "quadratic"
    settings["stop-training-time"] = "end"
    settings["training-skip-steps"] = 1
    settings["forcing"] = False

    settings["input-scale"] = "rms"
    # settings["input-scale"] = "none"

    # settings["truncation-type"] = "size"
    # settings["truncation-value"] = 5
    # settings["boundary-truncation-type"] = "size"
    # settings["boundary-truncation-value"] = 2

    settings["truncation-type"] = "energy"
    settings["truncation-value"] = 1 - 1e-5
    settings["boundary-truncation-type"] = "energy"
    settings["boundary-truncation-value"] = 1 - 1e-5

    settings["regularization-parameter"] = {
        "A": np.logspace(-4, 1, 6),
        "H": np.logspace(-4, 1, 6),
        "B": np.logspace(-4, 1, 6),
    }

    # settings['regularization-parameter'] =  1.0e-4
    # settings["regularization-parameter"] = [
    #     1.0e-10,
    #     1.0e-9,
    #     1.0e-8,
    #     1.0e-7,
    #     1.0e-6,
    #     1.0e-5,
    #     1.0e-4,
    #     1.0e-3,
    #     1.0e-2,
    #     1.0e-1,
    #     1.0e0,
    #     1.0e1,
    #     1.0e2,
    # ]
    # settings['trial-space-splitting-type'] = 'split'
    settings["trial-space-splitting-type"] = "combined"

    # settings['acceleration-computation-type'] = 'finite-difference'
    settings["acceleration-computation-type"] = "acceleration-snapshots"

    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    settings["model-name"] = "quadratic-opinf-2-multi-reg"
    normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
