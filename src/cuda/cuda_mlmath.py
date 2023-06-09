import cupy as cp

def relu(X):
    return cp.maximum(0, X)

def relu_derivative(x):
    return cp.where(x <= 0, 0, 1)

def sigmoid(x):
    return cp.divide(1, (1 + cp.exp(-x)))

def sigmoid_derivative(x):
    Z = sigmoid(x)
    return cp.multiply(Z, (1 - Z))

def mse(Y_hat, Y):
    return cp.mean((Y_hat - Y) ** 2)

def mse_derivative(output, y):
    # Calculate the error for each sample in the dataset
    error = output - y

    # Calculate the derivative of the loss function for each sample
    derivative = error / y.shape[0]

    return derivative

def binary_crossentropy(Y_hat, Y):
    eps = 1e-10
    m = Y.shape[0]
    cost = -1/m * cp.sum(Y * cp.log(Y_hat + eps) + (1-Y) * cp.log(1-Y_hat + eps))
    return cost

def binary_crossentropy_derivative(Y_hat, Y):
    eps = 1e-10
    m = Y.shape[0]
    return (Y_hat - Y) / (Y_hat * (1 - Y_hat) + eps) / m

def batch_softmax(x):
    # Stabilize by subtracting row max from each row
    row_maxes = cp.max(x, axis=1)
    row_maxes = row_maxes[:, cp.newaxis]  # for broadcasting
    stabilized_exp = cp.exp(x - row_maxes)

    return cp.divide(stabilized_exp, cp.sum(stabilized_exp, axis=1, keepdims=True))

def batch_error_softmax_input(Y_hat, Y):
    return Y_hat - Y.T

def cross_entropy(Y_hat, Y):
    # Efficient, but assumes y is one-hot
    return -cp.log(Y_hat[cp.where(Y.T)] + 1e-8)

