import os
import subprocess

import numpy as np
import yaml

MASTER_YAMLS = ["../torsion-1-master.yaml", "../torsion-2-master.yaml"]
PATH_TO_NORMA = "/home/andiaz/tpls/Norma.jl"
JULIA = "~/.juliaup/bin/julia"
POLYNOMIAL_TYPES = ["linear", "quadratic", "cubic"]


def main():
    problem_dir = "/".join(
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
    model_dir = (
        problem_dir
        + "dynamic-neohookean-hex-hex/"
        + "impl-expl-dt-2em6-1em6-DN-predictor/"
        + "rom/models/"
    )
    romtype = "quadratic"
    problem = "torsion"

    print("Submitting jobs for ROM-ROM predictive case")
    print(f"  ROM type = {romtype}", flush=True)

    # neglog_energy_cutoffs = [4]
    neglog_energy_cutoffs = [4, 6, 8]
    submitter = NormaJobSubmitter(
        problem,
        model_dir,
        romtype,
        working_dir="./",
    )
    submitter.create_all_jobs(neglog_energy_cutoffs)
    submitter.submit_all_jobs(concurrent_jobs=10)


class NormaJobSubmitter:
    def __init__(
        self,
        problem: str,
        model_dir: str,
        romtype: str,
        working_dir: str = "./",
    ):
        self.problem = problem
        self.model_dir = model_dir
        self.romtype = romtype
        self.working_dir = working_dir
        self.processes = []
        self.job_dirs = []

    def create_all_jobs(self, neglog_energy_cutoffs: list[float]):
        self.job_dirs = [
            self._create_job(
                nlec,
            )
            for nlec in neglog_energy_cutoffs
        ]

    def _create_job(self, nlec: float) -> str:
        job_id = f"{self.romtype}-etrunc-1em{nlec}"
        print(f"Creating job for {job_id}.....", end="", flush=True)
        job_dir = self.working_dir + job_id + "/"
        os.makedirs(job_dir, exist_ok=True)
        self._save_new_yaml(job_dir, self._generate_yaml_dict(job_id))
        print("done.")
        return job_dir

    def _save_new_yaml(self, job_id: str, yaml_dicts: list[dict]):
        for i, yaml_dict in enumerate(yaml_dicts):
            with open(job_id + f"{self.problem}-{i + 1}.yaml", "w") as file:
                yaml.dump(yaml_dict, file, sort_keys=False)

    def _generate_yaml_dict(self, job_id: str) -> list[dict[dict]]:
        yaml_dicts = []
        for i, master_yaml in enumerate(MASTER_YAMLS):
            with open(master_yaml, "r") as file:
                yaml_dict = yaml.safe_load(file)

            if self.romtype in POLYNOMIAL_TYPES:
                yaml_dict["model"]["type"] = self.romtype + " opinf rom"
            else:
                raise ValueError(f"{self.romtype} is not supported.")
            yaml_dict["model"]["model-file"] = (
                self.model_dir
                + "-".join(
                    [
                        self.problem,
                        str(i + 1),
                        job_id,
                    ]
                )
                + ".npz"
            )
            yaml_dicts.append(yaml_dict)

        return yaml_dicts

    def submit_all_jobs(self, concurrent_jobs: int = 10):
        self.processes = []
        for job_dir in self.job_dirs:
            self.processes.append(self._submit_job(job_dir))
            self._manage_queue(concurrent_jobs=concurrent_jobs)

    def _submit_job(self, job_dir: str):
        print(f"Submitting job {job_dir}.....", end="", flush=True)

        command = " ".join(
            [
                "cp",
                "../../torsion.yaml",
                ".;",
                JULIA,
                f"--project=@{PATH_TO_NORMA}",
                f"{PATH_TO_NORMA}/src/Norma.jl",
                "torsion.yaml",
            ]
        )

        with open(job_dir + "output.log", "w") as file:
            process = subprocess.Popen(
                command,
                cwd=job_dir,
                shell=True,
                stdout=file,
                stderr=file,
            )
        print("done.")
        return process

    def _manage_queue(self, concurrent_jobs: int):
        while len(self.processes) == concurrent_jobs:
            for process in self.processes:
                if process.poll() is not None:
                    self.processes.remove(process)


if __name__ == "__main__":
    main()
