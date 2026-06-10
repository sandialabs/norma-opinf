import sys
from collections import defaultdict
from itertools import product
from typing import Union

import normaopinf
import numpy as np

import opinf


def multi_regselect(
    opinf_model: opinf.models.ContinuousModel,
    x: np.ndarray,
    xdot: np.ndarray,
    xddot: np.ndarray,
    bcs: np.ndarray,
    times: np.ndarray,
    reg_candidates: dict[str, Union[list, np.ndarray]],
    forcing: bool,
    model_type: str,
):
    print("Performing grid search")
    xdim = x.shape[0]
    bcdim = bcs.shape[0]
    groups, reg_combos = get_reg_combos(reg_candidates)
    assert _reg_groups_are_permissible(groups, model_type, forcing=forcing)
    # Store errors for different regularization parameters
    errors = np.zeros(len(reg_combos))

    # Create an extension of the time window to run ROMs into the future
    extend_window_ratio = 1
    times_extended = times * 1
    dt = times[1] - times[0]
    n_steps = times.shape[0]
    for i in range(1, extend_window_ratio):
        time_window = times_extended[-1] + (times - times[0]) + dt
        times_extended = np.append(times_extended, time_window)
    assert np.allclose(times_extended[0 : times.size], times)

    # Loop over regularization parameters, fit, and test
    for counter, reg_combo in enumerate(reg_combos):
        opinf_model.solver = opinf.lstsq.TikhonovSolver(
            reg_factory(xdim, bcdim, groups, reg_combo)
        )

        # Fit
        opinf_model.fit(states=x, ddts=xddot, inputs=bcs)

        n_cases = x.shape[-2]
        errors[counter] = 0.0
        for i in range(0, n_cases):

            # Create wrapper for forward evaluation of the model
            if isinstance(
                opinf_model, normaopinf.opinf.models.ShaneNonParametricOpInfModel
            ):
                opInfForwardModel = normaopinf.opinf.models.LinearOpInfRom(
                    -opinf_model.get_stiffness_matrix(),
                    opinf_model.get_exogenous_input_matrix(),
                )

            elif isinstance(
                opinf_model,
                normaopinf.opinf.models.ShaneNonParametricQuadraticOpInfModel,
            ):
                opInfForwardModel = normaopinf.opinf.models.QuadraticOpInfRom(
                    -opinf_model.get_stiffness_matrix(),
                    opinf_model.get_exogenous_input_matrix(),
                    -opinf_model.get_quadratic_stiffness_matrix(),
                    opinf_model,
                )

            elif isinstance(
                opinf_model, normaopinf.opinf.models.ShaneNonParametricCubicOpInfModel
            ):
                opInfForwardModel = normaopinf.opinf.models.CubicOpInfRom(
                    -opinf_model.get_stiffness_matrix(),
                    opinf_model.get_exogenous_input_matrix(),
                    -opinf_model.get_quadratic_stiffness_matrix(),
                    -opinf_model.get_cubic_stiffness_matrix(),
                    opinf_model,
                )
            else:
                print("Model type not found, exiting")
                sys.exit()

            def bc_hook(step):
                step_to_get = min(step, bcs.shape[-1] - 1)
                return bcs[..., i, step_to_get]

            # Test forward simulation
            test_states = opInfForwardModel.advance_n_steps_newmark(
                x[..., i, 0],
                xdot[..., i, 0],
                xddot[..., i, 0],
                dt[i],
                int(n_steps * extend_window_ratio),
                bc_hook,
            )

            # Check if we blew up
            if np.any(np.isnan(test_states)) or np.any(np.abs(test_states) > 1e5):
                print("Detected NaN in solution")
                errors[counter] += 1e10
            else:
                local_error = np.linalg.norm(
                    test_states[:, 0 : times.shape[0]] - x[:, i, :]
                ) / np.linalg.norm(x[:, i, :])
                errors[counter] += local_error

        print(
            "Regularization Parameter:",
        )
        for g, val in zip(groups, reg_combo):
            print(f"  {g}: {val:1.4e}")
        print(f"  Error: {errors[counter]/n_cases:1.3e}\n")
        counter += 1
    optimal_case = np.nanargmin(errors)
    optimal_reg_combo = reg_combos[optimal_case]
    best_error = np.nanmin(errors)
    print("Best regularization parameters:")
    for g, val in zip(groups, optimal_reg_combo):
        print(f"  {g}: {val:1.4e}")
    print(f"  Error = {best_error:1.3e}")
    opinf_model.solver = opinf.lstsq.TikhonovSolver(
        reg_factory(xdim, bcdim, groups, optimal_reg_combo)
    )
    opinf_model.fit(states=x, ddts=xddot, inputs=bcs)
    optimal_reg_dict = {g: val for g, val in zip(groups, optimal_reg_combo)}
    return opinf_model, optimal_reg_dict


def get_reg_combos(reg_candidates: dict[str, Union[list, np.ndarray]]):
    groups = list(reg_candidates.keys())
    candidates = [regs for _, regs in reg_candidates.items()]
    assert _groups_are_disjoint(groups)
    return groups, list(product(*candidates))


def reg_factory(
    xdim: int, bcdim: int, groups: list[str], reg_combo: tuple[float]
) -> np.ndarray:
    assert len(groups) == len(reg_combo)
    reg_dict = defaultdict(float)
    for group, val in zip(groups, reg_combo):
        for operator in group:
            reg_dict[operator] = val

    reg_vec = []
    for operator in "cAHGB":
        if operator in reg_dict.keys():
            reg_vec.append(reg_dict[operator] * _get_onesvec(xdim, bcdim, operator))
    return np.concatenate(reg_vec)


def _get_onesvec(xdim: int, bcdim: int, operator: str) -> np.ndarray:
    if operator == "c":
        return np.ones(1)
    elif operator == "A":
        return np.ones(xdim)
    elif operator == "H":
        return np.ones(xdim * (xdim + 1) // 2)
    elif operator == "G":
        return np.ones(xdim * (xdim + 1) * (xdim + 2) // 6)
    elif operator == "B":
        return np.ones(bcdim)
    else:
        raise ValueError(f"{operator} is not a supported operator type.")


def _groups_are_disjoint(groups: list[str]) -> bool:
    joined_groups = "".join(groups)
    return len(joined_groups) == len(set(joined_groups))


def _reg_groups_are_permissible(
    groups: list[str],
    model_type: str,
    forcing: bool = False,
) -> bool:
    all_operators = "".join(groups)
    if model_type in ["linear", "linear-symmetric"]:
        allowed_operators = "AB"

    elif model_type == "quadratic":
        allowed_operators = "AHB"

    elif model_type == "cubic":
        allowed_operators = "AHGB"
    if forcing:
        allowed_operators = "c" + allowed_operators

    return set(all_operators) == set(allowed_operators)


# if __name__ == "__main__":
#     reg_candidates = {"cA": [1], "B": [2, 3], "H": [4, 5], "G": [6]}
#     groups, reg_combos = get_reg_combos(reg_candidates)
#     print(reg_combos)
#     for combo in reg_combos:
#         reg = reg_factory(2, 1, groups, combo)
#         print(reg)
