import cupy as np

class KernelInitializer():
    def __init__(self, input_size, output_size):
        self.i = input_size
        self.o = output_size

    def W(self):
        pass

    def B(self):
        pass


class UniformInitializer(KernelInitializer):
    def W(self):
        return np.random.uniform(0, 1, size=(self.i, self.o))

    def B(self):
        return np.zeros((1, self.o))

class XavierInitializer(KernelInitializer):

    def W(self):

        return np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (self.i + self.o)),
            size=(self.i, self.o)
        )

    def B(self):
        return np.zeros((1, self.o))

class KaimingHeInitializer(KernelInitializer):

    def W(self):
        return np.random.normal(
            loc=0,
            scale=np.sqrt(2 / self.i), # 2 / l -> variance, sqrt for std_dev
            size=(self.i, self.o)
        )

    def B(self):
        return np.zeros((1, self.o))


class SigmoidInitializer(KernelInitializer):

    def W(self):
        return np.random.normal(loc=0, scale=np.sqrt(6 / (self.i + self.o)), size=(self.i, self.o))

    def B(self):
        return np.zeros((1, self.o))
