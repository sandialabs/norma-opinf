import os
import subprocess

import numpy as np
import yaml

MASTER_FOM_YAML = "../../torsion-1-ND-master.yaml"
MASTER_ROM_YAML = "../torsion-2-ND-master.yaml"
# MASTER_FOM_YAML = "../torsion-1-master.yaml"
# MASTER_ROM_YAML = "../torsion-2-master.yaml"
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
    domain = "torsion-2"
    romtype = "quadratic"

    print("Submitting jobs for FOM-ROM predictive case")
    print(f"  ROM domain = {domain}", flush=True)
    print(f"  ROM type = {romtype}", flush=True)

    # neglog_energy_cutoffs = [4]
    neglog_energy_cutoffs = [4, 6]
    submitter = NormaJobSubmitter(
        model_dir,
        domain,
        romtype,
        working_dir="./neumann-dirichlet/",
    )
    submitter.create_all_jobs(neglog_energy_cutoffs)
    submitter.submit_all_jobs(concurrent_jobs=1)


class NormaJobSubmitter:
    def __init__(
        self,
        model_dir: str,
        domain: str,
        romtype: str,
        working_dir: str = "./",
    ):
        self.model_dir = model_dir
        self.domain = domain
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

    def _save_new_yaml(self, job_id: str, yaml_dict: dict):
        with open(job_id + f"{self.domain}.yaml", "w") as file:
            yaml.dump(yaml_dict, file, sort_keys=False)

    def _generate_yaml_dict(self, job_id: str) -> dict[dict]:
        with open(MASTER_ROM_YAML, "r") as file:
            yaml_dict = yaml.safe_load(file)

        if self.romtype in POLYNOMIAL_TYPES:
            yaml_dict["model"]["type"] = self.romtype + " opinf rom"
        else:
            raise ValueError(f"{self.romtype} is not supported.")
        yaml_dict["model"]["model-file"] = (
            self.model_dir + self.domain + "-" + job_id + ".npz"
        )

        return yaml_dict

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
                f"../{MASTER_FOM_YAML}",
                "torsion-1.yaml;",
                "cp",
                # "../../torsion.yaml",
                "../../../torsion.yaml",
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
