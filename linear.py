import numpy as np


class Linear():
    def __init__(self, model, activation_func, d_activation_func,
                 input_dim=None, output_dim=None, first_layer=False, last_layer=False,
                 learning_rate=0.001) -> None:
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        """
        A = W^TX
        Z = h(A) 
        """
        self.X = model.data
        self.A = None
        self.Z = None
        self.dW = None
        self.dA = None
        self.dZ = None

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.next_layer = None
        self.prev_layer = None

    def random_initialize(self):
        """
        Notice:
        output = w^Tx [(input_dim, output_dim) * (output)
        """
        self.W = np.random.randn(
            self.output_dim, self.input_dim) * np.sqrt(2 / self.input_dim)
        self.b = np.zeros(shape=(self.output_dim, 1))

    def backward_propagation(self):
        """
        Goal: calculate:
        1) dZ, dW, db, dA
        """
        if self.first_layer:
            prev_A = self.X
        else:
            prev_A = self.prev_layer.A

        if self.last_layer:
            next_dA = self.model.calculate_cost_derivative(self.A)
        else:
            next_dA = self.next_layer.dA

        m = prev_A.shape[1]

        self.dZ = next_dA * self.d_activation_func(self.Z)
        self.dW = self.dZ.dot(prev_A.T) / m
        self.db = np.sum(self.dZ, axis=1, keepdims=True) / m
        self.dA = self.W.T.dot(self.dZ)

    def update_weight(self):
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db
