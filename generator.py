from values import Value
from operations import Operations, Operation
from dataclasses import dataclass
from ml_types import Type
from printer import Printer
import random

RANDOM_SEED = 84
rng = random.Random(RANDOM_SEED)

help = Printer(rng) 

# Example Usage
# OperationInstance(
#     operation=Add,
#     args=[x0, x1]
# )

@dataclass
class OperationInstance:
    operation: Operation
    args: list[Value]

# Boolean check that the shapes of the arguments are compatible with the operation
def shapes_work(op: Operation, args: list[Value]) -> bool:
    """Check if the shapes of the arguments are compatible with the operation.

    Args:
        op (Operation): The operation to check.
        args (list[Value]): The arguments to the operation.

    Returns:
        bool: True if the shapes are compatible, False otherwise.
    """

    if op.name in {"Add", "Subtract"}:
        return args[0].shape == args[1].shape

    if op.name == "MatMul":
        # inner dimmensions must work
        return args[0].shape[1] == args[1].shape[0]

    if op.name == "Transpose":
        return args[0].shape is not None

    if op.name == "Sum":
        return args[0].shape is not None

    return False

def infer_output_shape(op: Operation, args: list[Value]) -> tuple[int, int] | None:
    if op.name in {"Add", "Subtract"}:
        return args[0].shape

    if op.name == "MatMul":
        return (args[0].shape[0], args[1].shape[1])

    if op.name == "Transpose":
        return (args[0].shape[1], args[0].shape[0])

    if op.name == "Sum":
        return None

    return None

def get_legal_operations(available_values: List[Value], current_len: int, n: int) -> List[OperationInstance]:
    """
    Return a list of operation instances that can be applied to available_values.
    - current_len: number of operations already in the sequence.
    - n: desired total sequence length (number of operations).
    Sum (scalar-producing) ops are only allowed when current_len == n-1 (the final step).
    """
    legal_ops: List[OperationInstance] = []
    force_scalar = (current_len == n - 1)

    # for final iteration, forcing to return scalar values for easy comparison
    # TODO: Get rid of this eventually. Want to transition to comparing intermediary states at this point 
    if force_scalar:
        candidate_ops = [op for op in Operations if op.output_type == Type.Scalar]
    else:
        candidate_ops = [op for op in Operations if op.output_type != Type.Scalar]

    for op in candidate_ops:
        # Prevent producing a scalar before the final step
        # TODO: Revisit. It might be fine to create scalars
        if op.output_type == Type.Scalar and not force_scalar:
            continue

        # unary operations: only needs one input
        if len(op.input_types) == 1:
            needed_type = op.input_types[0]
            for v in available_values:
                # TODO: Only works if matches needed type AND size
                if v.type == needed_type and shapes_work(op, [v]):
                    legal_ops.append(OperationInstance(op, [v]))

        # binary operations: needs two inputs, creates all pairs 
        elif len(op.input_types) == 2:
            needed_type_1 = op.input_types[0]
            needed_type_2 = op.input_types[1]
            for v1 in available_values:
                for v2 in available_values:
                    if v1.type == needed_type_1 and v2.type == needed_type_2 and shapes_work(op, [v1, v2]):
                        legal_ops.append(OperationInstance(op, [v1, v2]))
    
    return legal_ops


def create_compatible_shape_pool(rows, cols, rng, next_seed_id):
    """
    Create extra compatible matrix Values for a base shape (rows, cols). Attempts to minimize instances of calling (x0, x0) as both inputs. 
    Returns (new_values, next_seed_id).
    """

    matmul_cols = rng.randint(1, 5)
    matmul_rows = rng.randint(1, 5)

    compatible_shapes = [
        (rows, cols),           # for Add/Subtract
        (rows, cols),           # another same-shape value
        (cols, matmul_cols),    # right operand for MatMul: (rows, cols) @ (cols, k)
        (matmul_rows, rows),    # left operand for MatMul: (k, cols) @ (rows, cols)
    ]

    new_values = []
    for shape in compatible_shapes:
        new_values.append(Value(f"x{next_seed_id}", Type.Matrix, shape=shape))
        next_seed_id += 1

    return new_values, next_seed_id


# Creates the initial inputs
def instantiate_seed_matrices(count: int, rng: random.Random) -> List[Value]:
    seed_matrix_sizes = []
    next_seed_id = 0

    for _ in range(count):
        rows = rng.randint(1, 5)
        cols = rng.randint(1, 5)

        seed_matrix_sizes.append(Value(f"x{next_seed_id}", Type.Matrix, shape=(rows, cols)))
        next_seed_id += 1

        compatible_values, next_seed_id = create_compatible_shape_pool(rows, cols, rng, next_seed_id)
        seed_matrix_sizes.extend(compatible_values)

    return seed_matrix_sizes

# "Computes" the output after applying the operation instance 
def apply_operation(op_inst: OperationInstance, temp_index: int) -> Value:
    out_type = op_inst.operation.output_type
    out_shape = infer_output_shape(op_inst.operation, op_inst.args)
    name = f"t{temp_index}"
    return Value(name, out_type, out_shape)

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


