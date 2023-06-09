from linear import *
from loss_functions import *
from activation_functions import *


class MyModel():
    def __init__(self, architecture, input_size, loss_function, learning_rate=0.0075):
        self.learning_rate = learning_rate
        self.layers = self.create_architecture(architecture, input_size)
        # self.data = train_data
        # self.targets = train_targets

        self.loss_function, self.d_cost_function = loss_function[loss_function]

    def create_architecture(self, architecture, input_size):
        layers = []

        for index, arc in enumerate(architecture):
            print(index)
            if index == 0:
                input_dim = input_size
            else:
                input_dim = layers[-1].output_dim
            output_dim = arc["layer_size"]
            activation_func, d_activation_func = activation_functions[arc["activation"]]

            layer = Linear(self, activation_func, d_activation_func,
                           input_dim, output_dim, learning_rate=self.learning_rate)

            # set pointers
            if index != 0:
                layers[-1].next_layer = layer
                layer.prev_layer = layers[-1]
            layers.append(layer)

        # initial weight
        for layer in layers:
            layer.random_initialize()

        return layers

    def forward(self, x):
        self.data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].Z

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def optimize(self):
        for layer in self.layers:
            layer.optimize()

    def calculate_loss(self, y_hat, targets):
        return self.cost_function(y_hat, targets)

    def calculate_loss_derivative(self, y_hat, targets):
        return self.d_cost_function(y_hat, targets)

    def calculate_accuracy(self, test_data, test_targets):
        # Works for binary input right now
        self.forward_pass(test_data)

        y_hat = self.layers[-1].Z
        pred = np.where(y_hat > 0.5, 1, 0)

        return (pred == test_targets).mean()
