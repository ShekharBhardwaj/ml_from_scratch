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


if __name__ == "__main__":
    print("Testing Matrix class...")
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    print("m1 × m2:", m1.multiply_matrix(m2))
