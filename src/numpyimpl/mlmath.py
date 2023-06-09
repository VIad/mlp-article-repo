import numpy as np

def relu(X):
    return np.maximum(0, X)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    Z = sigmoid(x)
    return Z * (1 - Z)

def mse(Y_hat, Y):
    return np.mean((Y_hat - Y) ** 2)

def mse_derivative(output, y):
    # Calculate the error for each sample in the dataset
    error = output - y

    # Calculate the derivative of the loss function for each sample
    derivative = error / y.shape[0]

    return derivative

def binary_crossentropy(Y_hat, Y):
    eps = 1e-10
    m = Y.shape[0]
    cost = -1/m * np.sum(Y * np.log(Y_hat + eps) + (1-Y) * np.log(1-Y_hat + eps))
    return cost

def binary_crossentropy_derivative(Y_hat, Y):
    eps = 1e-10
    m = Y.shape[0]
    return (Y_hat - Y) / (Y_hat * (1 - Y_hat) + eps) / m

def batch_softmax(x):
    # Stabilize by subtracting row max from each row
    row_maxes = np.max(x, axis=1)
    row_maxes = row_maxes[:, np.newaxis]  # for broadcasting
    stabilized_exp = np.exp(x - row_maxes)

    return stabilized_exp / np.sum(stabilized_exp, axis=1, keepdims=True)

def batch_error_softmax_input(Y_hat, Y):
    return Y_hat - Y.T

def cross_entropy(Y_hat, Y):
    # Efficient, but assumes y is one-hot
    return -np.log(Y_hat[np.where(Y.T)] + 1e-8)

