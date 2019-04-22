import numpy as np


class PerceptronLayer:
    def __init__(self, in_count, out_count, weights: np.array = None):
        self.in_count = in_count
        self.out_count = out_count
        self.weights = np.array(weights).reshape(weights.shape[0], weights.shape[1])

    def forward(self, x: np.array):
        results_arr = np.zeros(shape=(self.out_count, 1))
        x = np.array(x).reshape((1, self.in_count))

        # Perform the multiply-add, row x col (i x j).
        for i in range(0, self.out_count):
            for j in range(0, self.in_count):
                results_arr[i] += x[0][j] * self.weights[i][j + 1]

        # Go through results and add bias weight, then check against zero.
        for i in range(self.out_count):
            # If we fail a test, set 0, otherwise 1;
            results_arr[i] = 0 if (results_arr[i] + self.weights[i][0]) <= 0 else 1

        # Return the results array.
        return np.array(results_arr)


class Sequential:
    def __init__(self, layers):
        self.multi_layers = layers
        pass

    def forward(self, x: np.array):
        result = x
        for layer in self.multi_layers:
            result = layer.forward(result)
        return result


class BooleanFactory():
    def create_AND(self):
        return PerceptronLayer(2, 1, np.array([[-30, 20, 20]]))

    def create_OR(self):
        return PerceptronLayer(2, 1, np.array([[-10, 20, 20]]))

    def create_NOT(self):
        return PerceptronLayer(1, 1, np.array([[10, -20]]))

    def create_NAND(self):
        return Sequential(np.array([
            PerceptronLayer(2, 1, np.array([[-30, 20, 20]])),
            PerceptronLayer(1, 1, np.array([[10, -20]]))
        ]))

    def create_NOR(self):
        return Sequential(np.array([
            PerceptronLayer(2, 1, np.array([[-10, 20, 20]])),
            PerceptronLayer(1, 1, np.array([[10, -20]]))
        ]))

    def create_XNOR(self):
        return Sequential(np.array([
                PerceptronLayer(2, 2, np.array([[-30, 20, 20], [10, -20, -20]])),
                PerceptronLayer(2, 1, np.array([[-10, 20, 20]]))
            ]))

    def create_XOR(self):
        return Sequential(np.array([
            PerceptronLayer(2, 2, np.array([[-30, 20, 20], [10, -20, -20]])),
            PerceptronLayer(2, 1, np.array([[10, -20, -20]]))
        ]))
