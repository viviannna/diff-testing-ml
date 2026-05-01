import numpy as np
from assumptions import assumption_checks


def tensor_to_numpy(val):
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
        a = np.array(tensor_to_numpy(torch_env[name]))
        b = np.array(tensor_to_numpy(tf_env[name]))

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
      x7:Matrix(6, 5)
      t1:Scalar
    """
    if v.shape is None:
        return f"{v.name}:{v.type.value}"
    return f"{v.name}:{v.type.value}{v.shape}"


def format_symbolic_op(op_inst, temp_name: str) -> str:
    op_name = op_inst.operation.name
    args = ", ".join(format_symbolic_value(arg) for arg in op_inst.args)
    return f"{op_name}({args}) -> {temp_name}"


def format_value_for_diff(val, max_chars: int = 800) -> str:
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


def values_equal(a, b, atol: float, rtol: float) -> tuple[bool, str]:
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

        status = "MATCH" if is_equal else "MISMATCH"
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

        if not is_equal:
            lines.append("")
            lines.append(red("Difference:") if use_color else "Difference:")
            lines.append(indent_text(reason, prefix="  "))

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