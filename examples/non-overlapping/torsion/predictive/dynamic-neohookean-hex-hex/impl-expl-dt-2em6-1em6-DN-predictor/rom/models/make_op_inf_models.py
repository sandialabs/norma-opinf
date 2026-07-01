# import os
import normaopinf
import normaopinf.opinf
import numpy as np


def main():

    data_dir = "/".join(
        [
            "/home",
            "andiaz",
            "tpls",
            "norma-opinf",
            "examples",
            "non-overlapping",
            "torsion",
            "predictive/",
        ]
    )
    config = (
        data_dir
        + "dynamic-neohookean-hex-hex/"
        + "impl-expl-dt-2em6-1em6-DN-predictor/"
    )
    model_type = "quadratic"
    domains_to_train = [1]
    energy_truncations = [4]
    # velo_vals = [500, 1000, 5000, 5500, 8000]
    velo_vals = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    settings = {}
    settings["training-data-directories"] = [config + f"a{vel}" for vel in velo_vals]

    settings["model-type"] = model_type
    settings["stop-training-time"] = "end"
    settings["training-skip-steps"] = 2
    settings["forcing"] = False

    settings["input-scale"] = "rms"
    # settings["input-scale"] = "none"

    # settings["regularization-parameter"] = {
    #     "A": np.logspace(-10, 0, 11),
    #     "B": np.logspace(-10, 0, 11),
    # }

    settings["regularization-parameter"] = {
        "A": np.logspace(-5, 0, 6),
        "H": np.logspace(-5, 0, 6),
        "B": np.logspace(-2, 3, 6),
    }

    # settings['trial-space-splitting-type'] = 'split'
    settings["trial-space-splitting-type"] = "combined"

    # settings['acceleration-computation-type'] = 'finite-difference'
    settings["acceleration-computation-type"] = "acceleration-snapshots"

    settings["boundary-truncation-type"] = "energy"
    settings["truncation-type"] = "energy"

    # settings["boundary-truncation-type"] = "size"
    # settings["boundary-truncation-value"] = 2
    # settings["truncation-type"] = "size"
    # settings["truncation-value"] = 5

    settings["boundary-truncation-value"] = 1 - 1e-5
    for domain in domains_to_train:
        settings["fom-yaml-file"] = config + f"a5500/torsion-{domain}.yaml"
        for et in energy_truncations:
            # settings["boundary-truncation-value"] = 1 - 10**-et
            settings["truncation-value"] = 1 - 10**-et
            settings["model-name"] = f"torsion-{domain}-{model_type}-etrunc-1em{et}"

            # print_settings(settings)

            snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
            normaopinf.opinf.make_opinf_model_from_snapshots_dict(
                snapshots_dict,
                settings,
            )


def print_settings(settings: dict):
    for key, val in settings.items():
        if isinstance(val, dict):
            print(f"{key}:")
            for kkey, vval in val.items():
                print(f"  {kkey}: {vval}")
        elif isinstance(val, list):
            print(f"{key}:")
            for el in val:
                print(f"  {el}")
        else:
            print(f"{key}: {val}")


def print_dict(rom_dict: dict):
    for key, val in rom_dict.items():
        print("  ", end="", flush=True)
        if isinstance(val, dict):
            print_dict(val)
        elif isinstance(val, list):
            print(f"{key}: list of length {len(val)}")
        elif isinstance(
            val,
            np.ndarray,
        ) and np.any([ax > 2 for ax in val.shape]):
            print(f"{key}: np.ndarray of shape {val.shape}")
        else:
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
    for domain in [1, 2]:
        for exponent in [4, 6, 8]:
            print(f"Energy cutoff = 1e-{exponent}")
            print_dict(
                np.load(
                    f"torsion-{domain}-quadratic-etrunc-1em{exponent}.npz",
                ),
            )
