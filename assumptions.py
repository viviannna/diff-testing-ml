import numpy as np
from dataclasses import dataclass
from typing import Callable

RED = "\033[97;41;1m"     # bold white text on red background
GREEN = "\033[97;42;1m"   # bold white text on green background
RESET = "\033[0m"


def red(s: str) -> str:
    return f"{RED} {s} {RESET}"


def green(s: str) -> str:
    return f"{GREEN} {s} {RESET}"


@dataclass
class AssumptionCheck:
    name: str
    fn: Callable[[list[np.ndarray]], bool | str]

def pass_fail(result: bool, use_color: bool = True) -> str:
    if result:
        return "PASS"
    return red("FAIL") if use_color else "FAIL"


def is_square(a: np.ndarray) -> bool:
    return a.ndim >= 2 and a.shape[-1] == a.shape[-2]


def is_floating_or_complex(a: np.ndarray) -> bool:
    return (
        np.issubdtype(a.dtype, np.floating)
        or np.issubdtype(a.dtype, np.complexfloating)
    )


def is_floating(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.floating)


def is_symmetric(a: np.ndarray, tol: float = 1e-8) -> bool:
    return is_square(a) and np.allclose(a, np.swapaxes(a, -1, -2), atol=tol, rtol=tol)


def is_positive_definite(a: np.ndarray) -> bool:
    if not is_square(a):
        return False
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


def det_value(a: np.ndarray):
    if not is_square(a):
        return None
    try:
        return float(np.linalg.det(a))
    except Exception:
        return None


def cond_value(a: np.ndarray):
    if not is_square(a):
        return None
    try:
        return float(np.linalg.cond(a))
    except Exception:
        return None


def is_well_conditioned(a: np.ndarray, max_cond: float = 1e8) -> bool:
    cond = cond_value(a)
    return cond is not None and np.isfinite(cond) and cond < max_cond
    


def run_assumption_group(
    checks: list[AssumptionCheck],
    args: list[np.ndarray],
    use_color: bool = True,
) -> list[str]:
    results = []

    for check in checks:
        try:
            result = check.fn(args)

            if isinstance(result, bool) or isinstance(result, np.bool_):
                results.append(f"{check.name}: {pass_fail(bool(result), use_color)}")
            else:
                results.append(f"{check.name}: {result}")

        except Exception as e:
            error_text = f"ERROR ({type(e).__name__}: {e})"
            if use_color:
                error_text = red(error_text)
            results.append(f"{check.name}: {error_text}")

    return results

def assumption_checks(op_name: str, op_inst, env, tensor_to_numpy_fn, use_color: bool = True):
    """
    Return:
      api_results_by_framework, implied_results

    api_results_by_framework format:
      {
        "torch": ["square matrix: PASS", ...],
        "tf": ["Hermitian/symmetric positive definite: FAIL", ...]
      }
    """
    if op_name not in ASSUMPTION_CHECKS:
        return {}, []

    args = [
        np.array(tensor_to_numpy_fn(env[arg.name]))
        for arg in op_inst.args
    ]

    op_checks = ASSUMPTION_CHECKS[op_name]

    api_results_by_framework = {}
    api_checks = op_checks.get("api", {})

    # New format: api is a dict by framework
    if isinstance(api_checks, dict):
        for framework_name, checks in api_checks.items():
            api_results_by_framework[framework_name] = run_assumption_group(checks, args, use_color=use_color)

    # Backward compatibility: api is just a list
    else:
        api_results_by_framework["common"] = run_assumption_group(api_checks, args, use_color=use_color)

    implied_results = run_assumption_group(op_checks.get("implied", []), args, use_color=use_color)

    return api_results_by_framework, implied_results
ASSUMPTION_CHECKS = {
    "LogDet": {
        "api": {
            "torch": [
                AssumptionCheck(
                    "square matrix (determinant is only defined for square matrices)",
                    lambda args: is_square(args[0]),
                ),
                AssumptionCheck(
                    "floating/complex dtype (logdet is a numerical linear algebra operation)",
                    lambda args: is_floating_or_complex(args[0]),
                ),
                AssumptionCheck(
                    "determinant nonzero for finite output (det(A)=0 makes log(det(A)) singular / -inf)",
                    lambda args: det_value(args[0]) is not None and det_value(args[0]) != 0.0,
                ),
                AssumptionCheck(
                    "determinant positive for finite real logdet (real log(x) requires x > 0)",
                    lambda args: det_value(args[0]) is not None and det_value(args[0]) > 0.0,
                ),
            ],
            "tf": [
                AssumptionCheck(
                    "square matrix (TensorFlow requires shape [..., M, M])",
                    lambda args: is_square(args[0]),
                ),
                AssumptionCheck(
                    "floating/complex dtype (TensorFlow only accepts float/complex matrix dtypes)",
                    lambda args: is_floating_or_complex(args[0]),
                ),
                AssumptionCheck(
                    "Hermitian/symmetric positive definite (TensorFlow logdet only supports HPD matrices)",
                    lambda args: is_symmetric(args[0]) and is_positive_definite(args[0]),
                ),
            ],
        },
        "implied": [
            AssumptionCheck(
                "determinant nonzero (zero determinant means the matrix is singular)",
                lambda args: det_value(args[0]) is not None and det_value(args[0]) != 0.0,
            ),
            AssumptionCheck(
                "determinant positive (needed for a finite real-valued log(det(A)))",
                lambda args: det_value(args[0]) is not None and det_value(args[0]) > 0.0,
            ),
            AssumptionCheck(
                "symmetric/Hermitian-real (relevant because HPD implies symmetry/Hermitian structure)",
                lambda args: is_symmetric(args[0]),
            ),
            AssumptionCheck(
                "positive definite (all eigenvalues must be positive for Cholesky/HPD assumptions)",
                lambda args: is_positive_definite(args[0]),
            ),
            AssumptionCheck(
                "well-conditioned, cond(A) < 1e8 (large condition number means numerical instability is more likely)",
                lambda args: is_well_conditioned(args[0]),
            ),
            AssumptionCheck(
                "det(A) (raw determinant used to interpret logdet behavior)",
                lambda args: f"{det_value(args[0])}",
            ),
            AssumptionCheck(
                "cond(A) (condition number used as a numerical stability diagnostic)",
                lambda args: f"{cond_value(args[0])}",
            ),
        ],
    },

    "Cholesky": {
        "api": {
            "torch": [
                AssumptionCheck(
                    "square matrix (Cholesky decomposes square matrices)",
                    lambda args: is_square(args[0]),
                ),
                AssumptionCheck(
                    "floating/complex dtype (Cholesky is defined over real/complex numerical matrices)",
                    lambda args: is_floating_or_complex(args[0]),
                ),
                AssumptionCheck(
                    "symmetric/Hermitian (Cholesky assumes real symmetric or complex Hermitian input)",
                    lambda args: is_symmetric(args[0]),
                ),
                AssumptionCheck(
                    "positive definite (Cholesky requires all eigenvalues to be positive)",
                    lambda args: is_positive_definite(args[0]),
                ),
            ],
            "tf": [
                AssumptionCheck(
                    "square matrix (TensorFlow Cholesky expects square inner matrix dimensions)",
                    lambda args: is_square(args[0]),
                ),
                AssumptionCheck(
                    "floating/complex dtype (Cholesky is a numerical factorization)",
                    lambda args: is_floating_or_complex(args[0]),
                ),
                AssumptionCheck(
                    "symmetric/Hermitian (TensorFlow Cholesky assumes symmetric/Hermitian input)",
                    lambda args: is_symmetric(args[0]),
                ),
                AssumptionCheck(
                    "positive definite (TensorFlow Cholesky requires positive definiteness)",
                    lambda args: is_positive_definite(args[0]),
                ),
            ],
        },
        "implied": [
            AssumptionCheck(
                "well-conditioned, cond(A) < 1e8 (ill-conditioned matrices can amplify numerical error)",
                lambda args: is_well_conditioned(args[0]),
            ),
            AssumptionCheck(
                "cond(A) (condition number used as a numerical stability diagnostic)",
                lambda args: f"{cond_value(args[0])}",
            ),
        ],
    },

    "Solve": {
        "api": {
            "torch": [
                AssumptionCheck(
                    "A square (linear solve expects A in AX=B to be square)",
                    lambda args: is_square(args[0]),
                ),
                AssumptionCheck(
                    "floating/complex dtype (linear solve is a numerical algebra operation)",
                    lambda args: is_floating_or_complex(args[0])
                    and is_floating_or_complex(args[1]),
                ),
                AssumptionCheck(
                    "compatible RHS (B must have matching row dimension for AX=B)",
                    lambda args: (
                        args[0].ndim >= 2
                        and args[1].ndim >= 2
                        and args[0].shape[-1] == args[1].shape[-2]
                    ),
                ),
                AssumptionCheck(
                    "A invertible / unique solution (solve assumes AX=B has a unique solution)",
                    lambda args: (
                        is_square(args[0])
                        and np.linalg.matrix_rank(args[0]) == args[0].shape[-1]
                    ),
                ),
            ],
            "tf": [
                AssumptionCheck(
                    "matrix shape [..., M, M] (TensorFlow solve expects square coefficient matrices)",
                    lambda args: is_square(args[0]),
                ),
                AssumptionCheck(
                    "rhs shape [..., M, K] (right-hand side must be matrix-like with matching M)",
                    lambda args: args[1].ndim >= 2,
                ),
                AssumptionCheck(
                    "matrix and rhs same dtype (TensorFlow requires both inputs to share dtype)",
                    lambda args: args[0].dtype == args[1].dtype,
                ),
                AssumptionCheck(
                    "compatible RHS (matrix * output must have the same shape as rhs)",
                    lambda args: (
                        args[0].ndim >= 2
                        and args[1].ndim >= 2
                        and args[0].shape[-1] == args[1].shape[-2]
                    ),
                ),
            ],
        },
        "implied": [
            AssumptionCheck(
                "A invertible/full rank (singular matrices do not have a unique solve result)",
                lambda args: (
                    is_square(args[0])
                    and np.linalg.matrix_rank(args[0]) == args[0].shape[-1]
                ),
            ),
            AssumptionCheck(
                "A well-conditioned, cond(A) < 1e8 (ill-conditioned solves can produce unstable answers)",
                lambda args: is_well_conditioned(args[0]),
            ),
            AssumptionCheck(
                "cond(A) (condition number used to judge solve stability)",
                lambda args: f"{cond_value(args[0])}",
            ),
        ],
    },

    "Softmax": {
        "api": {
            "torch": [
                AssumptionCheck(
                    "non-empty tensor (softmax needs values along the selected dimension)",
                    lambda args: args[0].size > 0,
                ),
                AssumptionCheck(
                    "floating dtype (softmax outputs probabilities from real-valued logits)",
                    lambda args: is_floating(args[0]),
                ),
                AssumptionCheck(
                    "axis/dim=-1 valid (the selected softmax dimension must exist)",
                    lambda args: args[0].ndim >= 1,
                ),
            ],
            "tf": [
                AssumptionCheck(
                    "non-empty tensor (softmax needs values along the selected axis)",
                    lambda args: args[0].size > 0,
                ),
                AssumptionCheck(
                    "floating dtype (TensorFlow softmax expects floating logits)",
                    lambda args: is_floating(args[0]),
                ),
                AssumptionCheck(
                    "axis=-1 valid (the selected softmax axis must exist)",
                    lambda args: args[0].ndim >= 1,
                ),
            ],
        },
        "implied": [
            AssumptionCheck(
                "finite logits (NaN/Inf logits can make probability outputs undefined or framework-specific)",
                lambda args: np.isfinite(args[0]).all(),
            ),
            AssumptionCheck(
                "does not contain NaN (NaNs usually propagate through softmax)",
                lambda args: not np.isnan(args[0]).any(),
            ),
            AssumptionCheck(
                "does not contain Inf (Inf logits can create ambiguous Inf/Inf normalization behavior)",
                lambda args: not np.isinf(args[0]).any(),
            ),
        ],
    },
}