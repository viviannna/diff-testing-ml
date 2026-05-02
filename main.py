from pathlib import Path
from datetime import datetime
from comparator import compare_envs, compare_steps
from generator import build_sequence
import random
from printer import Printer
from executor import SequenceExecutor, initialize_seed_arrays





def make_output_dir(name) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("outputs") / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# if __name__ == "__main__":


def main_loop(num_seed_values, seq_length, matrix_max_size, random_seed):

   

    RUN_NAME = f"seed_{random_seed}_seq_{seq_length}_max_{matrix_max_size}"


    
    rng = random.Random(random_seed)

    help = Printer(rng, random_seed) 
    output_dir = make_output_dir(RUN_NAME)

    # 1. SYMBOLIC EXECUTION 
    seed_values, ops_applied, values = build_sequence(
        num_seed_values=num_seed_values,
        seq_length=seq_length,
        rng=rng,
        max_size=matrix_max_size,
    )

    help.print_generated_seq(ops_applied, values, seed_values)
    symbolic_text = help.format_generated_seq(ops_applied, values, seed_values)
    (output_dir / "symbolic_exec.txt").write_text(symbolic_text)

    # 2. EXECUTE THE SEQUENCES 

    # generate shared starting matrices
    initial_arrays = initialize_seed_arrays(seed_values, rng_seed=84)

    # PyTorch Sequences
    print("Generating the PyTorch sequences")
    torch_exec = SequenceExecutor(seed_values, ops_applied, "torch", initial_arrays)
    (output_dir / "torch_init_values.txt").write_text(
        torch_exec.format_initial_values()
    )
    (output_dir / "torch_exec_trace.txt").write_text(
        torch_exec.format_execution_trace()
    )
    (output_dir / "torch_final_values.txt").write_text(
        torch_exec.format_final_env()
    )
    torch_env = torch_exec.execute()
    print("Done generating the PyTorch sequences")

    print("Generating the TensorFlow steps")
    tf_exec  = SequenceExecutor(seed_values, ops_applied, "tf", initial_arrays)

    (output_dir / "tf_init_values.txt").write_text(
        tf_exec.format_initial_values()
    )
    (output_dir / "tf_exec_trace.txt").write_text(
        tf_exec.format_execution_trace()
    )
    (output_dir / "tf_final_values.txt").write_text(
        tf_exec.format_final_env()
    )
    tf_env = tf_exec.execute()
    print("Done generating the TensorFlow steps")

    print("Running summary report")
    summary_report = compare_envs(torch_env, tf_env, num_seed_values, seq_length, random_seed, matrix_max_size, atol=1e-5, rtol=1e-5)
    print(summary_report)
    (output_dir / "summary.txt").write_text(summary_report)

    print(f"Wrote outputs to: {output_dir}")


    comparison_report = compare_steps(
    ops_applied=ops_applied,
    torch_exec=torch_exec,
    tf_exec=tf_exec,
    torch_env=torch_env,
    tf_env=tf_env,
    num_seed_values=num_seed_values,
    seq_length=seq_length,
    seed=random_seed,
    max_size=matrix_max_size,
    atol=1e-5,
    rtol=1e-5,
    use_color=True,
    )
    (output_dir / "comparison.txt").write_text(comparison_report)

    print("Done")



if __name__ == "__main__":

    for i in range(200):

        main_loop(num_seed_values=20, seq_length=10, matrix_max_size=32, random_seed=i)
        main_loop(num_seed_values=20, seq_length=20, matrix_max_size=32, random_seed=i)
        main_loop(num_seed_values=20, seq_length=30, matrix_max_size=32, random_seed=i)
        main_loop(num_seed_values=20, seq_length=40, matrix_max_size=32, random_seed=i)

