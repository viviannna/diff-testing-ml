import numpy as np

def tensor_to_numpy(val):
    if hasattr(val, "detach"):  # torch
        return val.detach().cpu().numpy()
    if hasattr(val, "numpy"):   # tf
        return val.numpy()
    return np.array(val)


def compare_envs(torch_env, tf_env, atol=1e-5, rtol=1e-5) -> str:
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
        a = tensor_to_numpy(torch_env[name])
        b = tensor_to_numpy(tf_env[name])

        a = np.array(a)
        b = np.array(b)

        if a.shape != b.shape:
            mismatches.append(
                f"{name}:\n  SHAPE MISMATCH\n  torch: {a.shape}\n  tf:    {b.shape}"
            )
            continue

        if np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            match_count += 1
        else:
            diff = np.abs(a - b)
            max_diff = np.max(diff)

            # find first mismatch index
            bad = ~np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            idx = tuple(np.argwhere(bad)[0]) if bad.any() else None

            torch_val = a[idx] if idx is not None else "N/A"
            tf_val = b[idx] if idx is not None else "N/A"

            mismatches.append(
                f"{name}:\n"
                f"  shape: {a.shape}\n"
                f"  max abs diff: {max_diff}\n"
                f"  first mismatch index: {idx}\n"
                f"  torch: {torch_val}\n"
                f"  tf:    {tf_val}"
            )

    # Summary
    lines.append("Summary")
    lines.append(f"  total vars: {total}")
    lines.append(f"  matches: {match_count}")
    lines.append(f"  mismatches: {len(mismatches)}")

    if mismatches:
        lines.append("\nMISMATCHES:")
        lines.extend(mismatches)
    else:
        lines.append("\nAll values match within tolerance.")

    return "\n".join(lines)