import numpy as np


def cross_entropy_sigmoid(y, y_hat):
    m = y.shape[1]

    # cost = np.sum(y*np.log(y_hat) + (1-y)*np.log(1 - y_hat)) / (-1*m)
    cost = (1./m) * (-np.dot(y, np.log(y_hat+0.001).T) -
                     np.dot(1-y, np.log(1-y_hat+0.001).T))
    # So that we have a real number at the end instead of a singleton; e.g. [[3]] => 3
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


def cross_entropy_softmax(y, y_hat):
    # y is a vector of dimension [1 x num_of_inputs]
    #  IT IS NOT ONE HOT VECTOR !!!
    batch_size, num_classes = y_hat.shape
    y_target = np.zeros((batch_size, num_classes))
    y_target[np.arange(batch_size), y] = 1

    return -np.sum(y_target * np.log(y_hat), axis=1)


def cross_entropy_sigmoid_derivative(y, y_hat):
    m = y.shape[1]
    return (-(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat)))


def cross_entropy_softmax_derivative(y, y_hat):
    # y is a vector of dimension [1 x num_of_inputs]
    #  IT IS NOT ONE HOT VECTOR !!!
    batch_size, num_classes = y_hat.shape
    y_target = np.zeros((batch_size, num_classes))
    y_target[np.arange(batch_size), y] = 1

    dloss = -y_target * (1 - y_hat)

    return dloss


def mean_squared(y, y_hat):
    return np.sum((y - y_hat)**2).squeeze() / (y_hat.shape[1]*2)


def d_mean_squared(y, y_hat):
    return (y_hat - y)


loss_functions = {"cross_entropy_sigmoid": (cross_entropy_sigmoid, cross_entropy_sigmoid_derivative),
                  "cross_entropy_softmax": (cross_entropy_softmax, cross_entropy_softmax_derivative),
                  "mean_squared": (mean_squared, d_mean_squared)
                  }
