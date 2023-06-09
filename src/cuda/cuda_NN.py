import cupy as cp


import cuda_mlmath
from cuda_initializers import XavierInitializer
from cuda_optimizers import SGD
import copy

import time


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.6f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


class Layer:
    def __init__(self, input_size, output_size, activation, kernel_initializer=None):
        self.kernel_initializer = kernel_initializer
        self.optimizer = None
        if kernel_initializer is None:
            self.kernel_initializer = XavierInitializer(input_size, output_size)

        self.activation = activation

        self.W = self.kernel_initializer.W()
        self.B = self.kernel_initializer.B()

    def backward(self, w_or_error, learning_rate, t):
        activation_error = self.compute_activation_error(w_or_error)

        weights_error = cp.multiply(cp.dot(self.input, activation_error), (1 / self.N))

        # Breaks CuDNN if we try to take the mean of (N, 1) matrix with respect to axis 0
        bias_error = cp.mean(activation_error, axis=0, keepdims=True) if activation_error.shape[1] != 1 else cp.mean(activation_error, keepdims=True)

        self.W, self.B = self.optimizer.step(
            learning_rate,
            weights_error,
            bias_error,
            self.W,
            self.B,
            t
        )

        # compute the error with respect to this layer
        error = cp.dot(activation_error, self.W.T)

        return error

    def compute_activation_error(self, w_or_error):
        if self.activation == 'relu':
            return cp.multiply(cuda_mlmath.relu_derivative(self.D), w_or_error)
        if self.activation == 'sigmoid':
            return cp.multiply(cuda_mlmath.sigmoid_derivative(self.D), w_or_error)
        if self.activation == 'softmax':
            return w_or_error
        else:
            return w_or_error

    def dense_pass(self, X):
        return cp.dot(X, self.W) + self.B

    def activation_pass(self, X):
        if self.activation == 'relu':
            return cuda_mlmath.relu(X)
        if self.activation == 'sigmoid':
            return cuda_mlmath.sigmoid(X)
        if self.activation == 'softmax':
            return cuda_mlmath.batch_softmax(X)
        else:
            return X  # linear

    def layer(self, X):
        self.input = X.T
        self.N = X.shape[0]
        D = self.dense_pass(X)
        A = self.activation_pass(D)
        self.D = D
        return A


class NN:
    def __init__(self, layers, loss='mse', optimizer=SGD()):
        self.loss = loss
        self.layers = layers
        self.optimizer = optimizer
        for layer in layers:
            layer.optimizer = copy.copy(optimizer)

    def compute_loss_derivative(self, Y_hat, Y):
        if self.loss == 'mse':
            return cuda_mlmath.mse_derivative(Y_hat, Y)
        if self.loss == 'binary_crossentropy':
            return cuda_mlmath.binary_crossentropy_derivative(Y_hat, Y)
        else:
            return cuda_mlmath.batch_error_softmax_input(Y_hat, Y)

    def compute_error(self, Y_hat, Y):
        if self.loss == 'mse':
            return cuda_mlmath.mse(Y_hat, Y)
        if self.loss == 'binary_crossentropy':
            return cuda_mlmath.binary_crossentropy(Y_hat, Y)
        else:
            return cuda_mlmath.cross_entropy(Y_hat, Y)

    def predict(self, X):
        x_pred = cp.array(X)
        output = x_pred
        for layer in self.layers:
            output = layer.layer(output)
        output = cp.asnumpy(output)
        return output

    def predict_(self, X):
        output = X
        for layer in self.layers:
            output = layer.layer(output)
        return output

    def sample(self, x_train, sample_size, y_train):
        indices = cp.random.choice(x_train.shape[0], sample_size, replace=False)

        sample_X = x_train[indices]
        sample_Y = y_train[indices]
        return sample_X, sample_Y

    def fit(self, x_train, y_train, epochs, learning_rate, sample_size, include_logs=True):
        x_train_cp = cp.array(x_train)
        y_train_cp = cp.array(y_train)

        n_weight_updates = int(cp.ceil(x_train_cp.shape[0] / sample_size))
        t = 1
        for i in range(epochs):
            err = []
            for j in range(n_weight_updates):
                # random sample for minibatch j
                sample_X, sample_Y = self.sample(x_train_cp, sample_size, y_train_cp)

                output = self.predict_(sample_X).T

                # calculate loss derivative
                error = self.compute_loss_derivative(output, sample_Y).T

                # backward propagation
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate, t)
                t += 1
                # calculate average error on all samples
                total_err = self.compute_error(output, sample_Y).T
                err.append(total_err)
            if include_logs:
                print('epoch %d/%d err: %f' % (i + 1, epochs, cp.mean(cp.array(err))))