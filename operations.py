# Operations we currently support. These are the GENERIC ML ops. 
from dataclasses import dataclass
from ml_types import Type

@dataclass
class Operation:
    name: str
    input_types: list[Type]
    output_type: Type

# Example Usage
# OperationInstance(
#     operation=Add,
#     args=[x0, x1]
# )

@dataclass
class OperationInstance:
    operation: Operation
    args: list[Value]

Operations = [
    Operation("Add", [Type.Matrix, Type.Matrix], Type.Matrix),
    Operation("Subtract", [Type.Matrix, Type.Matrix], Type.Matrix),
    Operation("MatMul", [Type.Matrix, Type.Matrix], Type.Matrix),
    Operation("Transpose", [Type.Matrix], Type.Matrix),
    Operation("Sum", [Type.Matrix], Type.Scalar),  # Useful for converting final answer into a scalar and comparing across frameworks
]