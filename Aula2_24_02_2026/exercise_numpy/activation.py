#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    