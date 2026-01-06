from derivative import gradient

def gradient_descent(f, start, learning_rate=0.1, steps=50):
    point = start.copy()
    
    for i in range(steps):
        grad = gradient(f, point)
        for j in range(len(point)):
            point[j] = point[j] - learning_rate * grad[j]
        
        # Print every 10 steps
        if i % 10 == 0:
            print(f"Step {i}: point={point}, f={f(point):.6f}")
    
    return point


if __name__ == "__main__":
    def sum_of_squares(p):
        return p[0]**2 + p[1]**2
    
    result = gradient_descent(sum_of_squares, [4.0, 3.0])
    print("Minimum at:", result)
    print("f at minimum:", sum_of_squares(result))