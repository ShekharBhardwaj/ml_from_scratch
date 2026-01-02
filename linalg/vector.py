import math
class Vector:
    def __init__(self, components):
        """Store the vector components as a list"""
        self.components = components
    
    def __repr__(self):
        """So we can print the vector nicely"""
        return f"Vector({self.components})"
    
    def add(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])
    
    def subtract(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])
    
    def scale(self, scalar):
        return Vector([a * scalar for a in self.components])
    
    def magnitude(self):
        """Return the length of the vector"""
        return math.sqrt(sum(a ** 2 for a in self.components))
    
    
    def dot(self, other):
        """Dot product of two vectors, return a number"""
        return sum(a * b for a, b in zip(self.components, other.components))

    def normalize(self):
        return Vector([a / self.magnitude() for a in self.components])


if __name__ == "__main__":
    v1 = Vector([3, 4]) 
    v2 = Vector([5, 6])
    print("dot:", v1.dot(v2))
    print("magnitude:", v1.magnitude())
    print("add:", v1.add(v2))
    print("subtract:", v1.subtract(v2))
    print("scale:", v1.scale(1/2))
    unit = v1.normalize()
    print("normalize:", unit)
    print("magnitude of unit:", unit.magnitude())
