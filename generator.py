from values import Value
from operations import Operations, Operation, OperationInstance
from ml_types import Type, MatrixTypes, MatrixInstance
import random


STRUCTURED_SQUARE_TYPES = {
    MatrixInstance.Symmetric,
    MatrixInstance.SPD,
    MatrixInstance.Singular,
    MatrixInstance.Diagonal,
    MatrixInstance.Orthogonal,
    MatrixInstance.IllConditioned,
}

# More Random than each individual structured type, so the generated pool
# contains both generic matrices and property-targeted matrices.
MATRIX_TYPE_WEIGHTS = {
    MatrixInstance.Random: 4.0,
    MatrixInstance.Symmetric: 1.0,
    MatrixInstance.SPD: 1.5,
    MatrixInstance.Singular: 1.0,
    MatrixInstance.Diagonal: 1.0,
    MatrixInstance.Orthogonal: 1.0,
    MatrixInstance.IllConditioned: 1.0,
}


def choose_matrix_type_for_shape(
    shape: tuple[int, int],
    rng: random.Random,
) -> MatrixInstance:
    """
    Pick a matrix annotation that is compatible with this shape.

    Structured matrix properties like SPD, Diagonal, Orthogonal, Singular,
    and IllConditioned only make sense for square matrices. Rectangular
    helper matrices are therefore always Random.
    """
    rows, cols = shape

    if rows != cols:
        return MatrixInstance.Random

    choices = list(MatrixTypes)
    weights = [MATRIX_TYPE_WEIGHTS.get(t, 1.0) for t in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


# Boolean check that the shapes of the arguments are compatible with the operation
def shapes_work(op: Operation, args: list[Value]) -> bool:
    """Check if the shapes of the arguments are compatible with the operation."""

    if op.name in {"Add", "Subtract"}:
        return args[0].shape == args[1].shape

    if op.name == "MatMul":
        return (
            args[0].shape is not None
            and args[1].shape is not None
            and args[0].shape[1] == args[1].shape[0]
        )

    if op.name == "Transpose":
        return args[0].shape is not None

    if op.name == "Sum":
        return args[0].shape is not None

    if op.name in {"LogDet", "Cholesky"}:
        return (
            args[0].shape is not None
            and args[0].shape[0] == args[0].shape[1]
        )

    if op.name == "Solve":
        A, b = args
        return (
            A.shape is not None
            and b.shape is not None
            and A.shape[0] == A.shape[1]
            and A.shape[0] == b.shape[0]
        )

    if op.name == "Softmax":
        return args[0].shape is not None

    # Unsupported ops in operations.py should not be generated yet.
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

    if op.name == "LogDet":
        return None

    if op.name == "Cholesky":
        return args[0].shape

    if op.name == "Solve":
        return args[1].shape

    if op.name == "Softmax":
        return args[0].shape

    return None


def value_is_intermediate(v: Value) -> bool:
    return v.name.startswith("t")


def weighted_choice_values(
    candidates: list[Value],
    rng: random.Random,
    t_bonus: float = 3.0,
) -> Value:
    weights = [
        1.0 + t_bonus if value_is_intermediate(v) else 1.0
        for v in candidates
    ]
    return rng.choices(candidates, weights=weights, k=1)[0]


def pair_weight(
    pair: tuple[Value, Value],
    t_bonus: float = 3.0,
    same_arg_penalty: float = 0.2,
) -> float:
    v1, v2 = pair
    num_t = int(value_is_intermediate(v1)) + int(value_is_intermediate(v2))

    weight = 1.0 + t_bonus * num_t

    if v1.name == v2.name:
        weight *= same_arg_penalty

    return weight


def weighted_choice_pairs(
    candidates: list[tuple[Value, Value]],
    rng: random.Random,
    t_bonus: float = 3.0,
) -> tuple[Value, Value]:
    weights = [pair_weight(pair, t_bonus) for pair in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def sample_operation_instance(
    available_values: list[Value],
    current_len: int,
    n: int,
    rng: random.Random,
    t_bonus: float = 3.0,
) -> OperationInstance | None:
    """
    Randomly select an operation first, then select valid arguments.
    Inputs are biased toward intermediate t-values.
    """

    candidate_ops = [op for op in Operations]

    while candidate_ops:
        op = rng.choice(candidate_ops)
        candidate_ops.remove(op)

        # Unary operation
        if len(op.input_types) == 1:
            needed_type = op.input_types[0]

            candidates = [
                v for v in available_values
                if v.type == needed_type and shapes_work(op, [v])
            ]

            if not candidates:
                continue

            arg = weighted_choice_values(candidates, rng, t_bonus)
            return OperationInstance(op, [arg])

        # Binary operation
        elif len(op.input_types) == 2:
            needed_type_1 = op.input_types[0]
            needed_type_2 = op.input_types[1]

            valid_pairs: list[tuple[Value, Value]] = []

            for v1 in available_values:
                if v1.type != needed_type_1:
                    continue

                for v2 in available_values:
                    if v2.type != needed_type_2:
                        continue

                    if shapes_work(op, [v1, v2]):
                        valid_pairs.append((v1, v2))

            if not valid_pairs:
                continue

            v1, v2 = weighted_choice_pairs(valid_pairs, rng, t_bonus)
            return OperationInstance(op, [v1, v2])

    return None


def create_compatible_shape_pool(
    rows: int,
    cols: int,
    rng: random.Random,
    next_seed_id: int,
    matrix_type: MatrixInstance,
) -> tuple[list[Value], int]:
    """
    Create extra compatible matrix Values for a base shape.

    Important: these helper matrices do NOT all inherit the base matrix type.
    Each helper gets its own compatible matrix type. This avoids runs where a
    whole x0-x4 block is Diagonal, a whole x5-x9 block is Orthogonal, etc.
    """

    new_values: list[Value] = []

    if matrix_type == MatrixInstance.Random:
        matmul_cols = rng.randint(1, max(1, cols))
        matmul_rows = rng.randint(1, max(1, rows))

        compatible_shapes = [
            (rows, cols),           # for Add/Subtract
            (rows, cols),           # another same-shape value
            (cols, matmul_cols),    # right operand for MatMul
            (matmul_rows, rows),    # left operand for MatMul
        ]

    elif matrix_type in STRUCTURED_SQUARE_TYPES:
        if rows != cols:
            raise ValueError(
                f"{matrix_type.value} matrix must be square, got {(rows, cols)}"
            )

        n = rows
        compatible_shapes = [
            (n, n),  # for Add/Subtract
            (n, n),  # another same-shape value
            (n, n),  # MatMul-compatible on the right
            (n, n),  # MatMul-compatible on the left
        ]

    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    for shape in compatible_shapes:
        helper_type = choose_matrix_type_for_shape(shape, rng)

        new_values.append(
            Value(
                name=f"x{next_seed_id}",
                type=Type.Matrix,
                shape=shape,
                matrix_type=helper_type,
            )
        )
        next_seed_id += 1

    return new_values, next_seed_id


def instantiate_seed_matrices(
    count: int,
    rng: random.Random,
    max_size: int,
) -> list[Value]:
    seed_matrix_sizes: list[Value] = []
    next_seed_id = 0

    for _ in range(count):
        matrix_type = choose_matrix_type_for_shape((1, 1), rng)

        if matrix_type == MatrixInstance.Random:
            rows = rng.randint(1, max_size)
            cols = rng.randint(1, max_size)

        elif matrix_type in STRUCTURED_SQUARE_TYPES:
            n = rng.randint(1, max_size)
            rows, cols = n, n

        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

        seed_matrix_sizes.append(
            Value(
                name=f"x{next_seed_id}",
                type=Type.Matrix,
                shape=(rows, cols),
                matrix_type=matrix_type,
            )
        )
        next_seed_id += 1

        compatible_values, next_seed_id = create_compatible_shape_pool(
            rows, cols, rng, next_seed_id, matrix_type
        )
        seed_matrix_sizes.extend(compatible_values)

    return seed_matrix_sizes


# "Computes" the symbolic output after applying the operation instance.
def apply_operation(op_inst: OperationInstance, temp_index: int) -> Value:
    out_type = op_inst.operation.output_type
    out_shape = infer_output_shape(op_inst.operation, op_inst.args)
    name = f"t{temp_index}"

    matrix_type = None

    if out_type == Type.Matrix:
        op_name = op_inst.operation.name

        if op_name == "Add":
            # SPD + SPD is SPD.
            if all(arg.matrix_type == MatrixInstance.SPD for arg in op_inst.args):
                matrix_type = MatrixInstance.SPD

            # Diagonal + Diagonal is Diagonal.
            elif all(arg.matrix_type == MatrixInstance.Diagonal for arg in op_inst.args):
                matrix_type = MatrixInstance.Diagonal

            # Symmetric-like + symmetric-like is symmetric.
            elif all(arg.matrix_type in {
                MatrixInstance.Symmetric,
                MatrixInstance.SPD,
                MatrixInstance.Diagonal,
                MatrixInstance.IllConditioned,
            } for arg in op_inst.args):
                matrix_type = MatrixInstance.Symmetric

            else:
                matrix_type = MatrixInstance.Random

        elif op_name == "Subtract":
            # Diagonal - Diagonal is Diagonal.
            if all(arg.matrix_type == MatrixInstance.Diagonal for arg in op_inst.args):
                matrix_type = MatrixInstance.Diagonal

            # Symmetric-like - symmetric-like is symmetric, but not necessarily SPD.
            elif all(arg.matrix_type in {
                MatrixInstance.Symmetric,
                MatrixInstance.SPD,
                MatrixInstance.Diagonal,
                MatrixInstance.IllConditioned,
            } for arg in op_inst.args):
                matrix_type = MatrixInstance.Symmetric

            else:
                matrix_type = MatrixInstance.Random

        elif op_name == "Transpose":
            matrix_type = op_inst.args[0].matrix_type

        elif op_name == "MatMul":
            a, b = op_inst.args

            # Diagonal @ Diagonal is Diagonal.
            if (
                a.matrix_type == MatrixInstance.Diagonal
                and b.matrix_type == MatrixInstance.Diagonal
            ):
                matrix_type = MatrixInstance.Diagonal

            # Orthogonal @ Orthogonal is Orthogonal.
            elif (
                a.matrix_type == MatrixInstance.Orthogonal
                and b.matrix_type == MatrixInstance.Orthogonal
            ):
                matrix_type = MatrixInstance.Orthogonal

            else:
                matrix_type = MatrixInstance.Random

        elif op_name == "Cholesky":
            # Cholesky output is triangular, but there is no Triangular type yet.
            matrix_type = MatrixInstance.Random

        elif op_name == "Solve":
            matrix_type = MatrixInstance.Random

        elif op_name == "Softmax":
            matrix_type = MatrixInstance.Random

        else:
            matrix_type = MatrixInstance.Random

    return Value(
        name=name,
        type=out_type,
        shape=out_shape,
        matrix_type=matrix_type,
    )


def build_sequence(
    num_seed_values: int,
    seq_length: int,
    rng: random.Random,
    max_size: int,
) -> tuple[list[Value], list[OperationInstance], list[Value]]:
    """
    Build a deterministic symbolic sequence.

    Returns:
      seed_values, ops_applied, all_values_after_execution
    """

    seed_values = instantiate_seed_matrices(num_seed_values, rng, max_size)
    values = list(seed_values)

    ops_applied: list[OperationInstance] = []
    current_len = 0
    next_temp_idx = 0

    while current_len < seq_length:
        op_inst = sample_operation_instance(
            available_values=values,
            current_len=current_len,
            n=seq_length,
            rng=rng,
            t_bonus=3.0,
        )

        if not op_inst:
            print(f"No legal operations available at step {current_len}. Stopping early.")
            break

        new_val = apply_operation(op_inst, next_temp_idx)

        values.append(new_val)
        ops_applied.append(op_inst)

        current_len += 1
        next_temp_idx += 1

    return seed_values, ops_applied, values
