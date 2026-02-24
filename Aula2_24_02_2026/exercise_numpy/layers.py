#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer (Layer):
    
    def __init__(self, n_units, input_shape = None):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        
    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
 
    def output_shape(self):
         return (self.n_units,) 
     
    def set_biases(self, new_biases):
        self.biases = new_biases
        
    def set_weigths(self, new_wmatrix):
        self.weights = new_wmatrix

    
