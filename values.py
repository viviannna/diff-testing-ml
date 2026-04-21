# Values are variables + types (ex: x0: Matrix) useful for only using variables that match expected input type
# Example usage: x0 = Value("x0", Matrix)
from dataclasses import dataclass
from ml_types import Type

@dataclass
class Value:
    name: str
    type: Type
    shape: tuple[int, int] | None = None # None for scalars, (row, col) for matricies
    
