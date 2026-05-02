from enum import Enum

class Type(Enum):
    Matrix = "Matrix"
    Scalar = "Scalar"

class MatrixInstance(Enum):
    Random = "Random"
    Symmetric = "Symmetric"
    SPD = "SPD"
    Singular = "Singular"
    Diagonal = "Diagonal"
    Orthogonal = "Orthogonal"
    IllConditioned = "IllConditioned"

MatrixTypes = [
    MatrixInstance.Random,
    MatrixInstance.Symmetric,
    MatrixInstance.SPD,
    MatrixInstance.Singular,
    MatrixInstance.Diagonal,
    MatrixInstance.Orthogonal,
    MatrixInstance.IllConditioned,
]