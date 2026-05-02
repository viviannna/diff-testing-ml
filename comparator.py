import numpy as np
from assumptions import assumption_checks
from executor import ExecutionCrash, is_execution_crash


def tensor_to_numpy(val):
    if is_execution_crash(val):
        raise TypeError("ExecutionCrash does not have a tensor value.")
    if hasattr(val, "detach"):  # torch
        return val.detach().cpu().numpy()
    if hasattr(val, "numpy"):   # tf
        return val.numpy()
    return np.array(val)


def compare_envs(
    torch_env,
    tf_env,
    num_seed_values,
    seq_length,
    seed,
    max_size,
    atol=1e-5,
    rtol=1e-5,
) -> str:
    torch_keys = set(torch_env.keys())
    tf_keys = set(tf_env.keys())

    lines = []
    mismatches = []

    if torch_keys != tf_keys:
        lines.append("Variable sets differ.")
        lines.append(f"Only in torch: {sorted(torch_keys - tf_keys)}")
        lines.append(f"Only in tf: {sorted(tf_keys - torch_keys)}")
        return "\n".join(lines)

    shared_keys = sorted(torch_keys, key=lambda x: (x[0], int(x[1:])))

    total = len(shared_keys)
    match_count = 0

    for name in shared_keys:
        torch_val_raw = torch_env[name]
        tf_val_raw = tf_env[name]

        torch_crashed = is_execution_crash(torch_val_raw)
        tf_crashed = is_execution_crash(tf_val_raw)

        if torch_crashed or tf_crashed:
            if torch_crashed and tf_crashed:
                match_count += 1
                continue

            mismatches.append(
                f"{name}:\n"
                f"  CRASH MISMATCH\n"
                f"  torch: {format_value_for_diff(torch_val_raw)}\n"
                f"  tf:    {format_value_for_diff(tf_val_raw)}"
            )
            continue

        a = np.array(tensor_to_numpy(torch_val_raw))
        b = np.array(tensor_to_numpy(tf_val_raw))

        if a.shape != b.shape:
            mismatches.append(
                f"{name}:\n"
                f"  SHAPE MISMATCH\n"
                f"  torch: {a.shape}\n"
                f"  tf:    {b.shape}"
            )
            continue

        if np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            match_count += 1
            continue

        diff = np.abs(a - b)

        try:
            max_diff = np.nanmax(diff)
        except ValueError:
            max_diff = "N/A"

        bad = ~np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)

        if bad.shape == ():
            idx = ()
            torch_val = a.item()
            tf_val = b.item()
        elif bad.any():
            idx = tuple(np.argwhere(bad)[0])
            torch_val = a[idx]
            tf_val = b[idx]
        else:
            idx = None
            torch_val = "N/A"
            tf_val = "N/A"

        mismatches.append(
            f"{name}:\n"
            f"  shape: {a.shape}\n"
            f"  max abs diff: {max_diff}\n"
            f"  first mismatch index: {idx}\n"
            f"  torch: {torch_val}\n"
            f"  tf:    {tf_val}"
        )

    lines.append("Summary")
    lines.append(f"  seed: {seed}")
    lines.append(f"  num seed values: {num_seed_values}")
    lines.append(f"  seq length: {seq_length}")
    lines.append(f"  max size: {max_size}")
    lines.append(f"  atol: {atol}")
    lines.append(f"  rtol: {rtol}")
    lines.append("")
    lines.append(f"  total vars: {total}")
    lines.append(f"  matches: {match_count}")
    lines.append(f"  mismatches: {len(mismatches)}")

    if mismatches:
        lines.append("\nMISMATCHES:")
        lines.extend(mismatches)
    else:
        lines.append(f"\nAll values match within tolerance atol={atol} and rtol={rtol}.")

    return "\n".join(lines)


# Formatting for detailed comparison

RED = "\033[97;41;1m"     # bold white text on red background
GREEN = "\033[97;42;1m"   # bold white text on green background
RESET = "\033[0m"


def red(s: str) -> str:
    return f"{RED} {s} {RESET}"


def green(s: str) -> str:
    return f"{GREEN} {s} {RESET}"


def indent_text(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def format_symbolic_value(v) -> str:
    """
    Example:
      x7:Matrix[SPD](6, 5)
      t1:Scalar
    """
    matrix_type = getattr(v, "matrix_type", None)
    matrix_type_str = f"[{matrix_type.value}]" if matrix_type is not None else ""
    if v.shape is None:
        return f"{v.name}:{v.type.value}{matrix_type_str}"
    return f"{v.name}:{v.type.value}{matrix_type_str}{v.shape}"


def format_symbolic_op(op_inst, temp_name: str) -> str:
    op_name = op_inst.operation.name
    args = ", ".join(format_symbolic_value(arg) for arg in op_inst.args)
    return f"{op_name}({args}) -> {temp_name}"


def format_value_for_diff(val, max_chars: int = 800) -> str:
    if is_execution_crash(val):
        return val.short()

    arr = np.array(tensor_to_numpy(val))

    if arr.shape == ():
        return str(arr.item())

    text = np.array2string(
        arr,
        precision=6,
        suppress_small=False,
        threshold=20,
        edgeitems=3,
    )

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    return text


def matrix_property_checks(val, cond_threshold: float = 1e8) -> list[str]:
    """
    Return concrete matrix properties for a runtime value.
    This is intentionally independent from the symbolic MatrixInstance annotation.
    """
    if is_execution_crash(val):
        return ["crashed: YES"]

    try:
        arr = np.array(tensor_to_numpy(val))
    except Exception as e:
        return [f"property check unavailable: {type(e).__name__}: {e}"]

    props: list[str] = []

    if arr.ndim < 2:
        return [f"not a matrix value (ndim={arr.ndim})"]

    rows, cols = arr.shape[-2], arr.shape[-1]
    is_square = rows == cols

    props.append(f"shape: {arr.shape}")
    props.append(f"square: {'YES' if is_square else 'NO'}")
    props.append(f"finite: {'YES' if np.isfinite(arr).all() else 'NO'}")
    props.append(f"contains NaN: {'YES' if np.isnan(arr).any() else 'NO'}")
    props.append(f"contains Inf: {'YES' if np.isinf(arr).any() else 'NO'}")

    if not is_square:
        return props

    try:
        symmetric = np.allclose(arr, np.swapaxes(arr, -1, -2), atol=1e-8, rtol=1e-8, equal_nan=True)
    except Exception:
        symmetric = False

    try:
        diagonal = np.allclose(arr, np.diag(np.diagonal(arr)), atol=1e-8, rtol=1e-8, equal_nan=True)
    except Exception:
        diagonal = False

    try:
        rank = np.linalg.matrix_rank(arr)
        singular = rank < rows
    except Exception:
        rank = None
        singular = None

    try:
        cond = float(np.linalg.cond(arr))
    except Exception:
        cond = None

    try:
        np.linalg.cholesky(arr)
        positive_definite = True
    except Exception:
        positive_definite = False

    try:
        eye = np.eye(rows, dtype=arr.dtype if np.issubdtype(arr.dtype, np.number) else float)
        orthogonal = np.allclose(arr.T @ arr, eye, atol=1e-8, rtol=1e-8, equal_nan=True)
    except Exception:
        orthogonal = False

    props.append(f"symmetric: {'YES' if symmetric else 'NO'}")
    props.append(f"diagonal: {'YES' if diagonal else 'NO'}")
    props.append(f"positive definite: {'YES' if positive_definite else 'NO'}")

    if singular is None:
        props.append("singular: UNKNOWN")
    else:
        props.append(f"singular: {'YES' if singular else 'NO'}")

    props.append(f"orthogonal: {'YES' if orthogonal else 'NO'}")

    if cond is None or not np.isfinite(cond):
        props.append("condition number: UNKNOWN/INF")
        props.append("ill-conditioned: UNKNOWN")
    else:
        props.append(f"condition number: {cond}")
        props.append(f"ill-conditioned (cond >= {cond_threshold:g}): {'YES' if cond >= cond_threshold else 'NO'}")

    if rank is None:
        props.append("rank: UNKNOWN")
    else:
        props.append(f"rank: {rank}")

    return props


def format_property_block(name: str, val, prefix: str = "  ") -> list[str]:
    lines = [f"{prefix}{name}:"]
    for prop in matrix_property_checks(val):
        lines.append(f"{prefix}  - {prop}")
    return lines


def values_equal(a, b, atol: float, rtol: float) -> tuple[bool, str]:
    a_crashed = is_execution_crash(a)
    b_crashed = is_execution_crash(b)

    if a_crashed or b_crashed:
        if a_crashed and b_crashed:
            return True, "both frameworks crashed"

        return (
            False,
            "only one framework crashed\n"
            f"torch: {format_value_for_diff(a)}\n"
            f"tf:    {format_value_for_diff(b)}"
        )

    a_np = np.array(tensor_to_numpy(a))
    b_np = np.array(tensor_to_numpy(b))

    if a_np.shape != b_np.shape:
        return False, f"shape mismatch: torch {a_np.shape}, tf {b_np.shape}"

    if np.allclose(a_np, b_np, atol=atol, rtol=rtol, equal_nan=True):
        return True, "values match"

    diff = np.abs(a_np - b_np)

    try:
        max_abs_diff = np.nanmax(diff)
    except ValueError:
        max_abs_diff = "N/A"

    bad = ~np.isclose(a_np, b_np, atol=atol, rtol=rtol, equal_nan=True)

    if bad.shape == ():
        idx = ()
        torch_val = a_np.item()
        tf_val = b_np.item()
    elif bad.any():
        idx = tuple(np.argwhere(bad)[0])
        torch_val = a_np[idx]
        tf_val = b_np[idx]
    else:
        idx = None
        torch_val = "N/A"
        tf_val = "N/A"

    return (
        False,
        f"max abs diff: {max_abs_diff}\n"
        f"first mismatch index: {idx}\n"
        f"torch: {torch_val}\n"
        f"tf:    {tf_val}"
    )


def compare_steps(
    ops_applied,
    torch_exec,
    tf_exec,
    torch_env,
    tf_env,
    num_seed_values,
    seq_length,
    seed,
    max_size,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    use_color: bool = True,
) -> str:
    lines = []
    lines.append("Line-by-line framework comparison")
    lines.append(f"atol={atol}, rtol={rtol}")
    lines.append(f"  seed: {seed}")
    lines.append(f"  num seed values: {num_seed_values}")
    lines.append(f"  seq length: {seq_length}")
    lines.append(f"  max size: {max_size}")
    lines.append("=" * 80)

    for step_idx, op_inst in enumerate(ops_applied):
        temp_name = f"t{step_idx}"
        op_name = op_inst.operation.name
        arg_names = [arg.name for arg in op_inst.args]

        symbolic = format_symbolic_op(op_inst, temp_name)

        torch_val = torch_env[temp_name]
        tf_val = tf_env[temp_name]

        is_equal, reason = values_equal(torch_val, tf_val, atol, rtol)

        both_crashed = is_execution_crash(torch_val) and is_execution_crash(tf_val)
        status = "BOTH CRASHED" if both_crashed else "MATCH" if is_equal else "MISMATCH"
        if use_color:
            status_text = green(status) if is_equal else red(status)
        else:
            status_text = status

        lines.append("")
        lines.append("-" * 80)
        lines.append(f"Step {step_idx}")
        lines.append(f"  Symbolic: {symbolic}")
        lines.append(f"  Result: {status_text}")
        lines.append("")

        lines.append("Torch")
        lines.append(f"  Concrete: {torch_exec._framework_call_str(op_name, arg_names)}")
        lines.append("  Value:")
        lines.append(indent_text(format_value_for_diff(torch_val), prefix="    "))

        lines.append("")
        lines.append("TensorFlow")
        lines.append(f"  Concrete: {tf_exec._framework_call_str(op_name, arg_names)}")
        lines.append("  Value:")
        lines.append(indent_text(format_value_for_diff(tf_val), prefix="    "))

        lines.append("")
        lines.append("Concrete matrix property checks:")
        lines.append("  Inputs:")
        for arg in op_inst.args:
            lines.extend(format_property_block(arg.name, torch_env[arg.name], prefix="    "))
        lines.append("  Outputs:")
        lines.extend(format_property_block(f"torch {temp_name}", torch_val, prefix="    "))
        lines.extend(format_property_block(f"tf {temp_name}", tf_val, prefix="    "))

        if not is_equal:
            lines.append("")
            lines.append(red("Difference:") if use_color else "Difference:")
            lines.append(indent_text(reason, prefix="  "))

            if any(is_execution_crash(torch_env[arg.name]) for arg in op_inst.args):
                api_checks_by_framework, implied_checks = {}, []
            else:
                api_checks_by_framework, implied_checks = assumption_checks(
                    op_name=op_name,
                    op_inst=op_inst,
                    env=torch_env,
                    tensor_to_numpy_fn=tensor_to_numpy,
                    use_color=use_color,
                )

            if api_checks_by_framework or implied_checks:
                lines.append("")
                lines.append("Assumption checks:")

                if api_checks_by_framework:
                    lines.append("  API-specified assumptions:")

                    for framework_name, checks in api_checks_by_framework.items():
                        pretty_name = "PyTorch" if framework_name == "torch" else "TensorFlow" if framework_name == "tf" else framework_name
                        lines.append(f"    {pretty_name}:")
                        for check in checks:
                            lines.append(f"      - {check}")

                if implied_checks:
                    lines.append("  Implied mathematical/numerical assumptions:")
                    for check in implied_checks:
                        lines.append(f"    - {check}")

    return "\n".join(lines)