import os
import subprocess

import yaml

YAML_DIR = "/home/andiaz/tpls/norma-opinf/examples/non-overlapping/torsion/predictive/"
MASTER_YAMLS = ["torsion-1-master.yaml", "torsion-2-master.yaml"]
PATH_TO_NORMA = "/home/andiaz/tpls/Norma.jl"
JULIA_EXEC = "~/.juliaup/bin/julia"


def main():
    # velo_vals = [500, 1000, 5000, 5500, 8000]
    velo_vals = [1000, 2000, 3000, 4000, 4500, 5000, 6000, 7000, 8000]

    concurrent_jobs = 5
    problem = "torsion"
    config = "./dynamic-neohookean-hex-hex/impl-expl-dt-2em6-1em6-DN-predictor/"
    submitter = NormaJobSubmitter(problem=problem, working_dir=config)
    submitter.create_all_jobs(velo_vals, overwrite=False)
    submitter.submit_all_jobs(concurrent_jobs=concurrent_jobs)


class NormaJobSubmitter:
    def __init__(self, problem: str, working_dir: str = "./"):
        self.problem = problem
        self.working_dir = working_dir
        self.processes = []
        self.job_ids = []
        self.model_dirs = []

        os.makedirs(self.working_dir, exist_ok=True)

    def create_all_jobs(self, velo_vals: list[int], overwrite: bool = False):
        self.job_ids = [
            self._create_job(
                velo,
                overwrite,
            )
            for velo in velo_vals
        ]
        self.job_ids = [j for j in self.job_ids if j is not None]

    def _create_job(self, velo: int, overwrite: bool) -> str:
        job_id = self.working_dir + f"a{velo:d}/"

        if os.path.exists(job_id) and not overwrite:
            print(f"Job for a={velo} exits. Skipping")
            return None
        else:
            print(f"Creating job for a={velo}.....", end="", flush=True)
            os.makedirs(job_id, exist_ok=True)
            for i, master in enumerate(MASTER_YAMLS):
                self._save_new_yaml(
                    job_id,
                    i,
                    self._generate_yaml_dict(velo, YAML_DIR + master),
                )
            print("done.")
        return job_id

    def _save_new_yaml(self, job_id: str, domain: int, yaml_dict: dict):
        with open(job_id + f"{self.problem}-{domain+1}.yaml", "w") as file:
            yaml.dump(yaml_dict, file, sort_keys=False)

    def _generate_yaml_dict(
        self,
        velo: int,
        master_yaml_filename: str,
    ) -> dict[dict]:
        with open(master_yaml_filename, "r") as file:
            yaml_dict = yaml.safe_load(file)

        yaml_dict["initial conditions"]["velocity"][0]["function"] = f"a={velo}; -a*y*z"
        yaml_dict["initial conditions"]["velocity"][1]["function"] = f"a={velo}; a*x*z"
        return yaml_dict

    def submit_all_jobs(self, concurrent_jobs: int = 10):
        self.processes = []
        for job_id in self.job_ids:
            self.processes.append(self._submit_job(job_id))
            self._manage_queue(concurrent_jobs=concurrent_jobs)

    def _submit_job(self, job_id: str):
        print(f"Submitting job {job_id}.....", end="", flush=True)

        command = " ".join(
            [
                "cp",
                YAML_DIR + f"{self.problem}.yaml",
                ".;",
                JULIA_EXEC,
                f"--project=@{PATH_TO_NORMA}",
                f"{PATH_TO_NORMA}/src/Norma.jl",
                f"{self.problem}.yaml",
            ]
        )

        with open(job_id + "output.log", "w") as file:
            process = subprocess.Popen(
                command,
                cwd=job_id,
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
