import numpy as np


class Linear():
    def __init__(self, model, activation_func, d_activation_func,
                 input_dim=None, output_dim=None,
                 learning_rate=0.001) -> None:
        # Basis attributes:
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
        self.next_layer = None
        self.prev_layer = None

        # Forward pass:
        self.A = None
        self.Z = None

        # Backward pass:
        self.dW = None
        self.db = None
        self.delta = None

    def random_initialize(self):
        """
        W: (input_dim, output_dim)
        b: (output_dim)
        """
        self.W = np.random.randn(
            self.input_dim, self.output_dim) * np.sqrt(2 / self.input_dim)
        self.b = np.zeros(shape=(self.output_dim))

    def forward(self):
        """
        a = w.T @ prev_z
        z = activate(a)
        """
        if self.prev_layer is None:
            prev_Z = self.model.data
        else:
            prev_Z = self.prev_layer.A

        self.A = prev_Z @ self.W + self.b.T
        self.Z = self.activation_func(self.A)

    def backward(self):
        """
        Update: dW, db, dA, dZ
        """
        if self.prev_layer is None:
            prev_Z = self.model.data
        else:
            prev_Z = self.prev_layer.Z

        if self.next_layer is None:
            next_delta = self.model.calculate_loss_derivative(self.A)
        else:
            next_delta = self.next_layer.delta

        m = prev_Z.shape[1]

        self.dW = prev_Z.T @ next_delta / m
        self.db = np.sum(next_delta, axis=0) / m
        self.delta = next_delta @ self.W.T * (1 - np.power(prev_Z, 2))

    def optimize(self):
        # SGD
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db
