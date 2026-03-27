
# Define the valid types (input and output) we are currently testing. Should be in the GENERIC form (ex: matrix not tesnor)
from enum import Enum

class Type(Enum):
    Matrix = "Matrix"
    Scalar = "Scalar"

    