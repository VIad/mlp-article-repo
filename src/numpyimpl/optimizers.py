import numpy as np


class Optimizer:

    def step(self, learning_rate, dw, db, w, b, t):
        pass


class Adam(Optimizer):
    def __init__(self, b1=0.9, b2=0.999, epsilon=1e-8):
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0

    def step(self, learning_rate, dw, db, w, b, t):
        self.m_dw = self.b1 * self.m_dw + (1 - self.b1) * dw
        self.v_dw = self.b2 * self.v_dw + (1 - self.b2) * dw ** 2

        self.m_db = self.b1 * self.m_db + (1 - self.b1) * db
        self.v_db = self.b2 * self.v_db + (1 - self.b2) * db ** 2

        mt_w_hat = self.m_dw / (1 - self.b1 ** t)
        mt_b_hat = self.m_db / (1 - self.b1 ** t)

        vt_w_hat = self.v_dw / (1 - self.b2 ** t)
        vt_b_hat = self.v_db / (1 - self.b2 ** t)

        n_w = w - learning_rate * mt_w_hat / (np.sqrt(vt_w_hat) + self.epsilon)
        n_b = b - learning_rate * mt_b_hat / (np.sqrt(vt_b_hat) + self.epsilon)

        return n_w, n_b


class SGD(Optimizer):

    def step(self, learning_rate, dw, db, w, b, t):
        n_w = w - learning_rate * dw
        n_b = b - learning_rate * db
        return n_w, n_b
