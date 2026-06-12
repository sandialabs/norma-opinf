import os
import sys

import numpy as np
from normaopinf.readers import load_displacement_csv_files


def main():
    path_to_examples = "/home/andiaz/tpls/norma-opinf/examples/non-overlapping/"
    problem = "torsion"

    fomdir = (
        path_to_examples
        + problem
        + "/a5500_5500/"
        + "fom-fom/with-output/dynamic-neohookean-fom-fom-hex-hex-equal-201-snap/impl-expl-dt-2em6-1em6-DN-predictor/relax1"
    )
    romdir = (
        path_to_examples
        + problem
        + "/a5500_5500/"
        + "fom-rom/with-output/dynamic-neohookean-fom-rom-hex-hex-equal-201-snap/impl-expl-dt-2em6-1em6-DN-predictor/relax1-cubic"
    )

    for i in [1, 2]:
        domain = f"{problem}-{i}"
        print(
            f"    Domain = {domain}",
        )
        print("      Loading FOM snapshots.....", end="", flush=True)
        with PrintSuppressor():
            fom_disps, _ = load_displacement_csv_files(fomdir, domain)
        print(f"done. shape = {fom_disps.shape}")

        print("      Loading ROM snapshots.....", end="", flush=True)
        with PrintSuppressor():
            rom_disps, _ = load_displacement_csv_files(romdir, domain)
        print(f"done. shape = {rom_disps.shape}")

        error = compute_frobenius_error(
            fom_disps[:, :, : rom_disps.shape[-1]], rom_disps, relative=True
        )
        print(f"      Relative L2-L2 ROM error = {error:1.4e}\n")


def compute_frobenius_error(
    fom_sol: np.ndarray, rom_sol: np.ndarray, relative: bool = True
) -> float:
    return _frobenius_error(
        _stack_components(fom_sol),
        _stack_components(rom_sol),
        relative=relative,
    )


def _frobenius_error(
    x: np.ndarray,
    y: np.ndarray,
    relative: bool = True,
) -> float:
    if relative:
        return np.linalg.norm(x - y) / np.linalg.norm(x)
    return np.linalg.norm(x - y)


def _stack_components(sol: np.ndarray):
    return np.vstack([s for s in sol])


class PrintSuppressor:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    main()
