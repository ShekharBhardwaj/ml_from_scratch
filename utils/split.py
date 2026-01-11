def train_test_split(X, y, test_size=0.2):
    """
    Split data into training and test sets.
    
    X: features (e.g., hours studied)
    y: labels (e.g., pass/fail)
    test_size: fraction for testing (0.2 = 20%)
    
    Returns: X_train, X_test, y_train, y_test
    """
    # Calculate split point
    n = len(X)
    test_count = int(n * test_size)
    train_count = n - test_count
    
    # Split the data
    X_train = X[:train_count]
    X_test = X[train_count:]
    y_train = y[:train_count]
    y_test = y[train_count:]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test data
    hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    passed = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
    X_train, X_test, y_train, y_test = train_test_split(hours, passed, test_size=0.2)
    
    print("Training set:")
    print("  X:", X_train)
    print("  y:", y_train)
    print("\nTest set:")
    print("  X:", X_test)
    print("  y:", y_test)