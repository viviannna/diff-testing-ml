from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Regexes for parsing summary.txt files
# -----------------------------------------------------------------------------

RUN_DIR_RE = re.compile(r"seed_(?P<seed>\d+)_seq_(?P<seq>\d+)_max_(?P<max>\d+)_")

INT_FIELDS = {
    "seed": re.compile(r"^\s*seed:\s*(\d+)", re.MULTILINE),
    "num_seed_values": re.compile(r"^\s*num seed values:\s*(\d+)", re.MULTILINE),
    "seq_length": re.compile(r"^\s*seq length:\s*(\d+)", re.MULTILINE),
    "max_size": re.compile(r"^\s*max size:\s*(\d+)", re.MULTILINE),
    "total_vars": re.compile(r"^\s*total vars:\s*(\d+)", re.MULTILINE),
    "matches": re.compile(r"^\s*matches:\s*(\d+)", re.MULTILINE),
    "mismatches": re.compile(r"^\s*mismatches:\s*(\d+)", re.MULTILINE),
}

# Example crash line:
# CRASH[torch] step=5 op=Cholesky(t0) _LinAlgError: ...
CRASH_RE = re.compile(
    r"CRASH\[(?P<framework>torch|tf)\]\s+"
    r"step=(?P<step>\d+)\s+"
    r"op=(?P<op>\w+)\((?P<args>.*?)\)\s+"
    r"(?P<error_type>\w+):\s*(?P<message>.*?)(?=\n\s*(?:torch:|tf:|t\d+:)|\Z)",
    re.DOTALL,
)

# Mismatch blocks start with t<number>: at the beginning of a line.
MISMATCH_START_RE = re.compile(r"^t\d+:", re.MULTILINE)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def extract_int(pattern: re.Pattern, text: str, default=None):
    match = pattern.search(text)
    return int(match.group(1)) if match else default


def safe_div(num, denom):
    if denom is None or denom == 0 or pd.isna(denom):
        return None
    return num / denom


def classify_crash(error_type: str | None, message: str | None) -> str:
    """
    Returns one of:
      - root: a direct framework crash at this operation
      - downstream: an UpstreamCrash caused by an earlier failed temp value
      - none: no crash
    """
    if error_type is None:
        return "none"

    message = message or ""
    if error_type == "UpstreamCrash" or "UpstreamCrash" in message:
        return "downstream"

    return "root"


# -----------------------------------------------------------------------------
# summary.txt parsing
# -----------------------------------------------------------------------------

def parse_summary_file(path: Path) -> tuple[dict, list[dict]]:
    text = path.read_text(errors="replace")

    run = {name: extract_int(pattern, text) for name, pattern in INT_FIELDS.items()}
    run["run_dir"] = str(path.parent)

    # Fallback to directory name if fields are missing for any reason.
    dir_match = RUN_DIR_RE.search(path.parent.name)
    if dir_match:
        run["seed"] = run["seed"] if run["seed"] is not None else int(dir_match.group("seed"))
        run["seq_length"] = run["seq_length"] if run["seq_length"] is not None else int(dir_match.group("seq"))
        run["max_size"] = run["max_size"] if run["max_size"] is not None else int(dir_match.group("max"))

    run["match_rate"] = safe_div(run.get("matches"), run.get("total_vars"))
    run["mismatch_rate"] = safe_div(run.get("mismatches"), run.get("total_vars"))

    mismatch_section = text.split("MISMATCHES:", 1)[1] if "MISMATCHES:" in text else ""
    mismatch_rows: list[dict] = []

    # Split mismatch section into blocks like:
    # t5:
    #   CRASH MISMATCH
    #   torch: ...
    #   tf: ...
    starts = list(MISMATCH_START_RE.finditer(mismatch_section))

    for i, start in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(mismatch_section)
        block = mismatch_section[start.start():end]
        temp_name = block.split(":", 1)[0].strip()

        crash_matches = list(CRASH_RE.finditer(block))

        if not crash_matches:
            # Numeric mismatch, shape mismatch, or any other non-crash mismatch.
            mismatch_rows.append(
                {
                    "run_dir": str(path.parent),
                    "seed": run["seed"],
                    "seq_length": run["seq_length"],
                    "max_size": run["max_size"],
                    "temp": temp_name,
                    "step": None,
                    "op": None,
                    "args": None,
                    "framework_crashed": None,
                    "error_type": None,
                    "message": None,
                    "crash_kind": "none",
                    "is_crash_mismatch": False,
                    "is_root_crash": False,
                    "is_downstream_crash": False,
                    "is_non_crash_mismatch": True,
                }
            )
            continue

        # A mismatch block can contain one crash line, or in rare cases two crash lines.
        # Record each crash line separately, then aggregate later.
        for crash in crash_matches:
            error_type = crash.group("error_type")
            message = " ".join(crash.group("message").split())
            crash_kind = classify_crash(error_type, message)

            mismatch_rows.append(
                {
                    "run_dir": str(path.parent),
                    "seed": run["seed"],
                    "seq_length": run["seq_length"],
                    "max_size": run["max_size"],
                    "temp": temp_name,
                    "step": int(crash.group("step")),
                    "op": crash.group("op"),
                    "args": crash.group("args"),
                    "framework_crashed": crash.group("framework"),
                    "error_type": error_type,
                    "message": message,
                    "crash_kind": crash_kind,
                    "is_crash_mismatch": True,
                    "is_root_crash": crash_kind == "root",
                    "is_downstream_crash": crash_kind == "downstream",
                    "is_non_crash_mismatch": False,
                }
            )

    # Count independent mismatch blocks, not crash lines. This avoids double-counting if
    # a single mismatch block ever has both frameworks crashing differently.
    if mismatch_rows:
        mismatch_df_tmp = pd.DataFrame(mismatch_rows)
        block_level = mismatch_df_tmp.groupby("temp", dropna=False).agg(
            has_root_crash=("is_root_crash", "any"),
            has_downstream_crash=("is_downstream_crash", "any"),
            has_crash=("is_crash_mismatch", "any"),
            has_non_crash=("is_non_crash_mismatch", "any"),
        )

        run["reported_mismatches"] = run.get("mismatches")
        run["parsed_mismatch_blocks"] = int(len(block_level))
        run["root_crash_mismatch_blocks"] = int(block_level["has_root_crash"].sum())
        run["downstream_crash_mismatch_blocks"] = int(
            (~block_level["has_root_crash"] & block_level["has_downstream_crash"]).sum()
        )
        run["any_downstream_crash_blocks"] = int(block_level["has_downstream_crash"].sum())
        run["non_crash_mismatch_blocks"] = int(block_level["has_non_crash"].sum())
    else:
        run["reported_mismatches"] = run.get("mismatches")
        run["parsed_mismatch_blocks"] = 0
        run["root_crash_mismatch_blocks"] = 0
        run["downstream_crash_mismatch_blocks"] = 0
        run["any_downstream_crash_blocks"] = 0
        run["non_crash_mismatch_blocks"] = 0

    run["root_crash_rate_per_var"] = safe_div(run["root_crash_mismatch_blocks"], run.get("total_vars"))
    run["downstream_crash_rate_per_var"] = safe_div(
        run["downstream_crash_mismatch_blocks"], run.get("total_vars")
    )
    run["non_crash_mismatch_rate_per_var"] = safe_div(
        run["non_crash_mismatch_blocks"], run.get("total_vars")
    )

    return run, mismatch_rows


def collect(outputs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict] = []
    mismatch_rows: list[dict] = []

    for summary_path in sorted(outputs_dir.glob("seed_*_seq_*_max_*/summary.txt")):
        run, rows = parse_summary_file(summary_path)
        run_rows.append(run)
        mismatch_rows.extend(rows)

    return pd.DataFrame(run_rows), pd.DataFrame(mismatch_rows)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def build_seq_length_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_df.groupby("seq_length", dropna=False)
        .agg(
            runs=("run_dir", "count"),
            total_vars=("total_vars", "sum"),
            matches=("matches", "sum"),
            reported_mismatches=("reported_mismatches", "sum"),
            parsed_mismatch_blocks=("parsed_mismatch_blocks", "sum"),
            root_crash_mismatch_blocks=("root_crash_mismatch_blocks", "sum"),
            downstream_crash_mismatch_blocks=("downstream_crash_mismatch_blocks", "sum"),
            any_downstream_crash_blocks=("any_downstream_crash_blocks", "sum"),
            non_crash_mismatch_blocks=("non_crash_mismatch_blocks", "sum"),
            avg_reported_mismatches_per_run=("reported_mismatches", "mean"),
            avg_mismatch_rate=("mismatch_rate", "mean"),
            avg_root_crash_rate_per_var=("root_crash_rate_per_var", "mean"),
            avg_downstream_crash_rate_per_var=("downstream_crash_rate_per_var", "mean"),
        )
        .reset_index()
    )


def build_op_summary(mismatch_df: pd.DataFrame) -> pd.DataFrame:
    if mismatch_df.empty:
        return pd.DataFrame()

    crash_df = mismatch_df[mismatch_df["is_crash_mismatch"] == True].copy()
    if crash_df.empty:
        return pd.DataFrame()

    summary = (
        crash_df.groupby("op", dropna=False)
        .agg(
            all_crash_rows=("op", "count"),
            root_crash_rows=("is_root_crash", "sum"),
            downstream_crash_rows=("is_downstream_crash", "sum"),
            torch_crash_rows=("framework_crashed", lambda s: (s == "torch").sum()),
            tf_crash_rows=("framework_crashed", lambda s: (s == "tf").sum()),
        )
        .reset_index()
    )

    # Count unique mismatch blocks per operation. For root-crash culprit analysis,
    # this is usually what you want in the report.
    root_blocks = (
        crash_df[crash_df["is_root_crash"] == True]
        .drop_duplicates(["run_dir", "temp", "op"])
        .groupby("op")
        .size()
        .rename("root_crash_mismatch_blocks")
    )

    downstream_blocks = (
        crash_df[crash_df["is_downstream_crash"] == True]
        .drop_duplicates(["run_dir", "temp", "op"])
        .groupby("op")
        .size()
        .rename("downstream_crash_mismatch_blocks")
    )

    summary = summary.merge(root_blocks, on="op", how="left")
    summary = summary.merge(downstream_blocks, on="op", how="left")
    summary["root_crash_mismatch_blocks"] = summary["root_crash_mismatch_blocks"].fillna(0).astype(int)
    summary["downstream_crash_mismatch_blocks"] = summary[
        "downstream_crash_mismatch_blocks"
    ].fillna(0).astype(int)

    summary = summary.sort_values(
        ["root_crash_mismatch_blocks", "downstream_crash_mismatch_blocks", "all_crash_rows"],
        ascending=False,
    )
    return summary


def build_framework_root_summary(mismatch_df: pd.DataFrame) -> pd.DataFrame:
    if mismatch_df.empty:
        return pd.DataFrame()

    root_df = mismatch_df[mismatch_df["is_root_crash"] == True].copy()
    if root_df.empty:
        return pd.DataFrame()

    return (
        root_df.groupby(["op", "framework_crashed"], dropna=False)
        .size()
        .reset_index(name="root_crash_count")
        .sort_values(["root_crash_count", "op"], ascending=[False, True])
    )


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_plots(
    run_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    seq_summary: pd.DataFrame,
    op_summary: pd.DataFrame,
    framework_root_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Average total mismatch rate by sequence length.
    plt.figure()
    plt.bar(seq_summary["seq_length"].astype(str), seq_summary["avg_mismatch_rate"])
    plt.xlabel("Sequence length")
    plt.ylabel("Average mismatch rate")
    plt.title("Mismatch rate by sequence length")
    save_plot(out_dir / "mismatch_rate_by_seq_length.png")

    # 2. Average root crash rate by sequence length.
    plt.figure()
    plt.bar(
        seq_summary["seq_length"].astype(str),
        seq_summary["avg_root_crash_rate_per_var"],
    )
    plt.xlabel("Sequence length")
    plt.ylabel("Average root crash rate per compared variable")
    plt.title("Independent crash-divergence rate by sequence length")
    save_plot(out_dir / "root_crash_rate_by_seq_length.png")

    # 3. Root vs downstream mismatch blocks by sequence length.
    labels = seq_summary["seq_length"].astype(str)
    root = seq_summary["root_crash_mismatch_blocks"]
    downstream = seq_summary["downstream_crash_mismatch_blocks"]
    non_crash = seq_summary["non_crash_mismatch_blocks"]

    plt.figure()
    plt.bar(labels, root, label="Root crash mismatches")
    plt.bar(labels, downstream, bottom=root, label="Downstream crash mismatches")
    plt.bar(labels, non_crash, bottom=root + downstream, label="Non-crash mismatches")
    plt.xlabel("Sequence length")
    plt.ylabel("Mismatch blocks")
    plt.title("Mismatch categories by sequence length")
    plt.legend()
    save_plot(out_dir / "mismatch_categories_by_seq_length.png")

    # 4. Mismatches per run boxplot.
    seq_lengths = sorted(run_df["seq_length"].dropna().unique())
    data = [
        run_df.loc[run_df["seq_length"] == seq_len, "reported_mismatches"].dropna().values
        for seq_len in seq_lengths
    ]
    if data:
        plt.figure()
        plt.boxplot(data, tick_labels=[str(x) for x in seq_lengths])
        plt.xlabel("Sequence length")
        plt.ylabel("Reported mismatches per run")
        plt.title("Mismatch distribution by sequence length")
        save_plot(out_dir / "mismatch_distribution_by_seq_length.png")

    # 5. Root crash culprits by operation.
    if not op_summary.empty:
        op_root = op_summary.sort_values("root_crash_mismatch_blocks", ascending=False)
        plt.figure()
        plt.bar(op_root["op"].astype(str), op_root["root_crash_mismatch_blocks"])
        plt.xlabel("Operation")
        plt.ylabel("Root crash mismatch blocks")
        plt.title("Independent crash-divergence culprits by operation")
        plt.xticks(rotation=45, ha="right")
        save_plot(out_dir / "root_crash_culprits_by_operation.png")

        op_downstream = op_summary.sort_values("downstream_crash_mismatch_blocks", ascending=False)
        plt.figure()
        plt.bar(op_downstream["op"].astype(str), op_downstream["downstream_crash_mismatch_blocks"])
        plt.xlabel("Operation")
        plt.ylabel("Downstream crash mismatch blocks")
        plt.title("Downstream crash propagation by operation")
        plt.xticks(rotation=45, ha="right")
        save_plot(out_dir / "downstream_crashes_by_operation.png")

    # 6. Which framework crashed first, by operation.
    if not framework_root_summary.empty:
        pivot = framework_root_summary.pivot_table(
            index="op",
            columns="framework_crashed",
            values="root_crash_count",
            aggfunc="sum",
            fill_value=0,
        )
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        pivot.plot(kind="bar")
        plt.xlabel("Operation")
        plt.ylabel("Root crash count")
        plt.title("Which framework crashed first by operation")
        plt.xticks(rotation=45, ha="right")
        save_plot(out_dir / "root_crashes_by_framework_and_operation.png")


# -----------------------------------------------------------------------------
# Console report
# -----------------------------------------------------------------------------

def print_report(
    run_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    seq_summary: pd.DataFrame,
    op_summary: pd.DataFrame,
    framework_root_summary: pd.DataFrame,
) -> None:
    print("\n=== Overall ===")
    print(f"Runs found: {len(run_df)}")
    print(f"Total vars compared: {int(run_df['total_vars'].sum())}")
    print(f"Reported mismatches: {int(run_df['reported_mismatches'].sum())}")
    print(f"Root crash mismatch blocks: {int(run_df['root_crash_mismatch_blocks'].sum())}")
    print(f"Downstream-only crash mismatch blocks: {int(run_df['downstream_crash_mismatch_blocks'].sum())}")
    print(f"Non-crash mismatch blocks: {int(run_df['non_crash_mismatch_blocks'].sum())}")

    print("\n=== Sequence length summary ===")
    cols = [
        "seq_length",
        "runs",
        "reported_mismatches",
        "root_crash_mismatch_blocks",
        "downstream_crash_mismatch_blocks",
        "non_crash_mismatch_blocks",
        "avg_mismatch_rate",
        "avg_root_crash_rate_per_var",
    ]
    print(seq_summary[cols].to_string(index=False))

    print("\n=== Independent root-crash culprits by operation ===")
    if op_summary.empty:
        print("No crash mismatches found.")
    else:
        cols = [
            "op",
            "root_crash_mismatch_blocks",
            "downstream_crash_mismatch_blocks",
            "torch_crash_rows",
            "tf_crash_rows",
        ]
        print(op_summary[cols].to_string(index=False))

    print("\n=== Which framework crashed first? Root crashes only ===")
    if framework_root_summary.empty:
        print("No root crashes found.")
    else:
        print(framework_root_summary.to_string(index=False))

    print("\n=== Interpretation note ===")
    print(
        "Use root_crash_mismatch_blocks as the main independent finding count. "
        "Downstream crash mismatches are still useful for showing propagation, "
        "but they should not be presented as independent bugs."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    outputs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis_results")

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Could not find outputs directory: {outputs_dir}")

    run_df, mismatch_df = collect(outputs_dir)

    if run_df.empty:
        print(f"No summary.txt files found under {outputs_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    seq_summary = build_seq_length_summary(run_df)
    op_summary = build_op_summary(mismatch_df)
    framework_root_summary = build_framework_root_summary(mismatch_df)

    run_df.to_csv(out_dir / "run_summary.csv", index=False)
    mismatch_df.to_csv(out_dir / "mismatch_summary.csv", index=False)
    seq_summary.to_csv(out_dir / "seq_length_summary.csv", index=False)
    op_summary.to_csv(out_dir / "op_crash_summary.csv", index=False)
    framework_root_summary.to_csv(out_dir / "framework_root_crash_summary.csv", index=False)

    make_plots(
        run_df=run_df,
        mismatch_df=mismatch_df,
        seq_summary=seq_summary,
        op_summary=op_summary,
        framework_root_summary=framework_root_summary,
        out_dir=out_dir,
    )

    print_report(
        run_df=run_df,
        mismatch_df=mismatch_df,
        seq_summary=seq_summary,
        op_summary=op_summary,
        framework_root_summary=framework_root_summary,
    )

    print(f"\nWrote CSVs and plots to: {out_dir}")


if __name__ == "__main__":
    main()
