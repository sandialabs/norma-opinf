import re
from collections import defaultdict

import numpy as np

EXAMPLE_DIRECTORY = "/".join(
    [
        "/home",
        "andiaz",
        "tpls",
        "norma-opinf",
        "examples",
        "non-overlapping",
        "torsion",
        "predictive",
        "dynamic-neohookean-hex-hex",
        "impl-expl-dt-2em6-1em6-DN-predictor",
        "rom/",
    ]
)


def main():
    problem = "torsion"
    configs = ["fom-rom", "rom-fom", "rom-rom"]

    romtypes = ["quadratic"]
    energy_criteria = [4, 6]

    for config in configs:
        print(config)
        runs_dir = EXAMPLE_DIRECTORY + f"{config}/"
        for romtype in romtypes:
            iter_counts = defaultdict(list)
            print(f"ROM type = {romtype}:", flush=True)
            for nlec in energy_criteria:
                run_dir = runs_dir + f"{romtype}-etrunc-1em{nlec}"
                avg_iter, max_iter = count_schwarz_iterations(problem, run_dir)
                iter_counts["average"].append(avg_iter)
                iter_counts["max"].append(max_iter)
                print(
                    f"  ec=1e-{nlec}:",
                    f" avg iter = {avg_iter:1.4f},",
                    f" max iter = {max_iter:3d}",
                )

            print("  Saving counts.....", end="", flush=True)
            np.savez(
                runs_dir + f"{romtype}_schwarz_iter_counts",
                **iter_counts,
            )
            print("done.", flush=True)


def count_schwarz_iterations(problem: str, run_dir: str) -> tuple[float]:
    counts = []
    with open(run_dir + f"/{problem}.log", "r") as f:
        for line in f.readlines():
            if "Performed" in line:
                count = int(re.findall(r"\d+", line)[0])
                counts.append(count)
    return np.mean(counts), np.max(counts)


def test_load(filename):
    counts = np.load(filename)
    print("  Saved average counts: ", counts["average"])
    print("  Saved max counts: ", counts["max"])


if __name__ == "__main__":
    main()
