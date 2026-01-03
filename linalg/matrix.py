from vector import Vector

class Matrix:
    def __init__(self, rows):
       self.rows = rows
    
    def __repr__(self):
        return f"Matrix({self.rows})"
    
    def shape(self):
        return len(self.rows), len(self.rows[0])

    
    def get_column(self, index):
        return Vector([ row[index] for row in self.rows])

    
    def multiply_vector(self, vec):

        return Vector([Vector(row).dot(vec) for row in self.rows])

    def multiply_matrix(self, other):
        return Matrix([[Vector(row).dot(other.get_column(i)) for i in range(other.shape()[1])] for row in self.rows])
    

    def minor(self, row, col):
        """Return a new matrix with the given row and column removed"""
        return Matrix([
            [self.rows[i][j] for j in range(len(self.rows[i])) if j != col]
            for i in range(len(self.rows)) if i != row
        ])

    def cofactor(self, row, col):
        return ((-1) ** (row + col)) * self.minor(row, col).determinant()

    def determinant(self):
        if self.shape()[0] != self.shape()[1]:
            raise ValueError("Matrix must be square to calculate determinant")
        if self.shape()[0] == 1:
            return self.rows[0][0]
        if self.shape()[0] == 2:
            return self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]
        return sum(self.rows[0][i] * self.cofactor(0, i) for i in range(self.shape()[1]))


if __name__ == "__main__":
    print("Testing Matrix class...")
    m3 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print("det 3x3:", m3.determinant())  # Should be 0

    m4 = Matrix([[2, 1, 3], [1, 0, 2], [0, 1, 1]])
    print("det 3x3:", m4.determinant())  # Should be -1