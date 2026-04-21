from values import Value
from operations import Operations, Operation
from dataclasses import dataclass
from ml_types import Type
from helpers import Helpers
import random

RANDOM_SEED = 84
rng = random.Random(RANDOM_SEED)

help = Helpers(rng) 

# Example Usage
# OperationInstance(
#     operation=Add,
#     args=[x0, x1]
# )

@dataclass
class OperationInstance:
    operation: Operation
    args: list[Value]

def get_legal_operations(available_values: List[Value], current_len: int, n: int) -> List[OperationInstance]:
    """
    Return a list of operation instances that can be applied to available_values.
    - current_len: number of operations already in the sequence.
    - n: desired total sequence length (number of operations).
    Sum (scalar-producing) ops are only allowed when current_len == n-1 (the final step).
    """
    legal_ops: List[OperationInstance] = []
    force_scalar = (current_len == n - 1)

    # for first iteration, forcing to return scalar values for easy comparison
    if force_scalar:
        candidate_ops = [op for op in Operations if op.output_type == Type.Scalar]
    else:
        candidate_ops = [op for op in Operations if op.output_type != Type.Scalar]

    for op in candidate_ops:
        # Prevent producing a scalar before the final step
        if op.output_type == Type.Scalar and not force_scalar:
            continue

        # unary operations: only needs one input
        if len(op.input_types) == 1:
            needed_type = op.input_types[0]
            for v in available_values:
                if v.type == needed_type:
                    legal_ops.append(OperationInstance(op, [v]))

        # binary operations: needs two inputs, creates all pairs 
        elif len(op.input_types) == 2:
            needed_type_1 = op.input_types[0]
            needed_type_2 = op.input_types[1]
            for v1 in available_values:
                for v2 in available_values:
                    if v1.type == needed_type_1 and v2.type == needed_type_2:
                        legal_ops.append(OperationInstance(op, [v1, v2]))

    return legal_ops


# Creates the initial inputs
def instantiate_seed_matrices(count: int, rng: random.Random) -> List[Value]:
    """Create initial matrix Values named x0, x1, ..."""
    return [Value(f"x{i}", Type.Matrix) for i in range(count)]

# "Computes" the output after applying the operation instance 
def apply_operation(op_inst: OperationInstance, temp_index: int) -> Value:
    """
    Create a new Value representing the output of the operation instance.
    Names temps t0, t1, ...
    """
    out_type = op_inst.operation.output_type
    name = f"t{temp_index}"
    return Value(name, out_type)

def build_sequence(num_seed_values: int, seq_length: int, rng: random.Random) -> Tuple[List[OperationInstance], List[Value]]:
    """
    Build a deterministic sequence of seq_len operations.
    - seed_values: initial Values (should include matrices).
    - n: desired number of operations.
    Returns (operations_applied, all_values_after_execution).
    Note: chooses first legal op each step; replace selection logic as needed.
    """

    seed_values = instantiate_seed_matrices(num_seed_values, rng)
    values = list(seed_values) # values that are currently usable

    ops_applied: List[OperationInstance] = [] # history of all operations applied 
    current_len = 0
    next_temp_idx = 0 # Used to uniquely identify created variables

    while current_len < seq_length:

        # choose from legal_ops and apply to create a new value
        legal_ops = get_legal_operations(values, current_len, seq_length)
        if not legal_ops:
            print(f"No legal operations available at step {current_len}. Stopping early.")
            break  # stuck; no legal op available
        op_inst = rng.choice(legal_ops)  
        new_val = apply_operation(op_inst, next_temp_idx)

        # print_step_decisions(current_len, values, legal_ops, op_inst)

        # update historic trackers
        values.append(new_val) # output is a new value type
        ops_applied.append(op_inst)

        current_len += 1
        next_temp_idx += 1

    help.print_generated_seq(ops_applied, values, seed_values)
    return ops_applied, values


# TODO: Init random folder name based off time and just pipe the print statements there
build_sequence(num_seed_values=3, seq_length=5, rng=rng)


