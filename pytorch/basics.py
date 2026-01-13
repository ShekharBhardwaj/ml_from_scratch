import torch

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Create a simple tensor (PyTorch's version of an array)
x = torch.tensor([1, 2, 3])
print(f"Tensor: {x}")


# ============================================================
# TENSORS: PyTorch's version of arrays
# ============================================================

# A 1D tensor (like our Vector)
vector = torch.tensor([1.0, 2.0, 3.0])
print("1D Tensor (vector):")
print(vector)
print(f"Shape: {vector.shape}")  # Shape tells us the dimensions

print()

# A 2D tensor (like our Matrix)
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
print("2D Tensor (matrix):")
print(matrix)
print(f"Shape: {matrix.shape}")  # 2 rows, 3 columns


print("\n" + "="*50)
print("TENSOR OPERATIONS")
print("="*50)

# ============================================================
# Addition (like our Vector.add)
# ============================================================
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# In our code: a.add(b)
# In PyTorch: a + b
result = a + b
print(f"\nAddition: {a} + {b} = {result}")

# ============================================================
# Scaling (like our Vector.scale)
# ============================================================
# In our code: a.scale(2)
# In PyTorch: a * 2
result = a * 2
print(f"Scaling: {a} * 2 = {result}")

# ============================================================
# Dot product (like our Vector.dot)
# ============================================================
# In our code: a.dot(b)
# In PyTorch: torch.dot(a, b)
result = torch.dot(a, b)
print(f"Dot product: {a} · {b} = {result}")

# ============================================================
# Matrix multiplication (like our Matrix.multiply_vector)
# ============================================================
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
vector = torch.tensor([5.0, 6.0])

# In our code: matrix.multiply_vector(vector)
# In PyTorch: matrix @ vector (or torch.matmul)
result = matrix @ vector
print(f"\nMatrix @ vector:")
print(f"{matrix}")
print(f"@ {vector}")
print(f"= {result}")


print("\n" + "="*50)
print("AUTOMATIC GRADIENTS")
print("="*50)

# ============================================================
# The magic: requires_grad=True
# ============================================================

# Create a tensor that tracks gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"\nx = {x}")

# Do some computation
y = x[0]**2 + x[1]**2  # y = x0² + x1² = 4 + 9 = 13
print(f"y = x[0]² + x[1]² = {y}")

# Compute gradients automatically!
y.backward()

# The gradient is stored in x.grad
print(f"Gradient: {x.grad}")
print(f"Expected: [2*x0, 2*x1] = [4, 6]")