def accuracy(y_true, y_pred):
    """Proportion of correct predictions"""
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

def precision(y_true, y_pred):
    """Of all positive predictions, how many were correct?"""
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(y_true, y_pred):
    """Of all actual positives, how many did we catch?"""
    # You implement this
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall"""
    # You implement this
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


if __name__ == "__main__":
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 1, 1, 0, 0]
    
    print("Accuracy:", accuracy(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))