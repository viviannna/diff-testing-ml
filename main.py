from pathlib import Path
from datetime import datetime
from comparator import compare_envs
from generator import build_sequence
import random
from printer import Printer
from executor import SequenceExecutor, initialize_seed_arrays





def make_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("outputs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":


    RANDOM_SEED = 84
    rng = random.Random(RANDOM_SEED)

    help = Printer(rng, RANDOM_SEED) 
    output_dir = make_output_dir()

    # 1. SYMBOLIC EXECUTION 

    seed_values, ops_applied, values = build_sequence(
        num_seed_values=3,
        seq_length=5,
        rng=rng,
    )
    # help.print_generated_seq(ops_applied, values, seed_values)
    symbolic_text = help.format_generated_seq(ops_applied, values, seed_values)
    (output_dir / "symbolic_exec.txt").write_text(symbolic_text)

    

    # 2. EXECUTE THE SEQUENCES 

    initial_arrays = initialize_seed_arrays(seed_values, rng_seed=84)


    torch_exec = SequenceExecutor(seed_values, ops_applied, "torch", initial_arrays)
    tf_exec    = SequenceExecutor(seed_values, ops_applied, "tf", initial_arrays)

    # PyTorch



    (output_dir / "torch_init_values.txt").write_text(
        torch_exec.format_initial_values()
    )
    (output_dir / "torch_exec_trace.txt").write_text(
        torch_exec.format_execution_trace()
    )
    (output_dir / "torch_final_values.txt").write_text(
        torch_exec.format_final_env()
    )

    (output_dir / "tf_init_values.txt").write_text(
        tf_exec.format_initial_values()
    )
    (output_dir / "tf_exec_trace.txt").write_text(
        tf_exec.format_execution_trace()
    )
    (output_dir / "tf_final_values.txt").write_text(
        tf_exec.format_final_env()
    )

    torch_env = torch_exec.execute()
    tf_env = tf_exec.execute()



    
    comparison_report = compare_envs(torch_env, tf_env)
    print(comparison_report)
    (output_dir / "comparison.txt").write_text(comparison_report)

    print(f"Wrote outputs to: {output_dir}")


