import numpy as np
from numpy.typing import ArrayLike
import math
from random import random
from numba import jit


class Layer:
    def __init__(self, size: int, next_size: int):
        self.size = size
        self.neurons = np.zeros(size, dtype=np.float32)
        self.biases = np.zeros(size, dtype=np.float32)
        self.weights = np.zeros((size, next_size), dtype=np.float32)


class NeuralNetwork:
    @jit(forceobj=True)
    def __init__(self, learning_rate: float, activation, derivative, sizes: list[int]):
        self.learning_rate = learning_rate
        self.activation = activation
        self.derivative = derivative
        self.layers = []
        for i in range(len(sizes)):
            next_size = sizes[i + 1] if i != len(sizes) - 1 else 0
            self.layers.append(Layer(sizes[i], next_size))
            for j in range(sizes[i]):
                self.layers[i].biases[j] = random() * 2 - 1
                for k in range(next_size):
                    self.layers[i].weights[j][k] = random() * 2 - 1

    @jit(forceobj=True)
    def feed_forward(self, inputs: ArrayLike) -> ArrayLike:
        np.copyto(self.layers[0].neurons, inputs)
        for i in range(1, len(self.layers)):
            l = self.layers[i - 1]
            l1 = self.layers[i]
            for j in range(l1.size):
                l1.neurons[j] = 0
                for k in range(l.size):
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j]
                l1.neurons[j] += l1.biases[j]
                l1.neurons[j] = self.activation(l1.neurons[j])
        return self.layers[-1].neurons

    @jit(forceobj=True)
    def backpropagation(self, targets: ArrayLike) -> None:
        errors = np.array([targets[i] - self.layers[-1].neurons[i]
                           for i in range(self.layers[-1].size)])
        for k in range(len(self.layers) - 2, -1, -1):
            l = self.layers[k]
            l1 = self.layers[k + 1]
            errors_next = np.zeros(l.size, dtype=np.float32)
            gradients = np.zeros(l1.size, dtype=np.float32)
            for i in range(l1.size):
                gradients[i] = errors[i] * self.derivative(self.layers[k + 1].neurons[i])
                gradients[i] *= self.learning_rate
            deltas = np.zeros((l1.size, l.size), dtype=np.float32)
            for i in range(l1.size):
                for j in range(l.size):
                    deltas[i][j] = gradients[i] * l.neurons[j]
            for i in range(l.size):
                for j in range(l1.size):
                    errors_next[i] += l.weights[i][j] * errors[j]
            errors = np.zeros(l.size, dtype=np.float32)
            np.copyto(errors, errors_next)
            weights_new = np.zeros((len(l.weights), len(l.weights[0])), dtype=np.float32)
            for i in range(l1.size):
                for j in range(l.size):
                    weights_new[j][i] = l.weights[j][i] + deltas[i][j]
            l.weights = weights_new
            l1.biases += gradients
