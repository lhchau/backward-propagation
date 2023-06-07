# Implementation for Backward propagation

> Give 2 fully connected layers, implement back-propagation for MNIST dataset

- Input: (32, 32, 1) => Output: (10)

## Backward propagation

![backward propagation src PRML Bishop](./asset/back-propagation.png)

- Cross Entropy Loss L = $\sum_i t_i ln(y_i) + (1-t_i) ln(1-y_i)$
- Denote: $E = \text{Expected Loss}$
- $y_k = h(a_k) = h(\sum_i w_{ki}^Tz_i)$

_Derivative of $E_n$ with respect to weight $w_{ji}$:\_

$$
\frac{\partial E_n}{\partial w_{ji}} = \frac{\partial L}{\partial a_j} \frac{\partial a_j}{\partial w_{ji}} = \delta_j z_i
$$

where

$$
\delta_j = \frac{\partial E_n}{\partial a_{j}}
$$

_For output layer_

$$
\delta_k = y_k - t_k
$$

_For hidden layer_

$$
\delta_j = \frac{\partial E_n}{\partial a_{j}} = \sum_k  \frac{\partial E_n}{\partial a_{k}}  \frac{\partial a_{k}}{\partial a_{j}} = h'(a_j) \sum_k w_{kj} \delta_k
$$

where

$$
a_k \sim w_{kj} h(a_j)
$$
