import numpy as np


def main():
    problem = "torsion"
    romtype = "quadratic"
    domains = [1, 2]
    neglog_energy_criteria = [4, 6]

    for domain in domains:
        print(f"Domain = {domain}")
        for nlec in neglog_energy_criteria:
            print(f"Energy = 1-1e-{nlec}")
            model_name = "-".join(
                [problem, str(domain), romtype, "etrunc", f"1em{nlec:d}.npz"]
            )
            model = np.load(model_name)
            print_dict(model)


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
