# Values are variables + types (ex: x0: Matrix) useful for only using variables that match expected input type
# Example usage: x0 = Value("x0", Matrix)
from dataclasses import dataclass
from ml_types import Type, MatrixInstance

@dataclass
@dataclass
class Value:
    name: str
    type: Type
    matrix_type: MatrixInstance # optional in case scalar
    shape: tuple[int, int] | None = None