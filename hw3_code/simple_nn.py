import numpy as np

def ReLU(x, deriv=False):
	if deriv: 
		positives = np.maximum(0, x)
		positives[positives > 0] = 1
		return positives
	return np.maximum(0,x)

def identity(x, deriv = False):
	return x

class Layer:
	def __init__(self, num_nodes, activation_fn = identity):
		self.layer_dim = num_nodes
		self.activation_fn = activation_fn
		self.initialize_values()

	def initialize_values(self):
		self.values = self.activation_fn(np.zeros((self.layer_dim,)))

	def get_values(self):
		return self.values

	def set_values(self, values):
		self.values = values

	def get_dim(self):
		return self.layer_dim

	def set_dim(self, dim):
		self.layer_dim = dim

	def get_activation(self):
		return self.activation_fn

	def set_activation(self, fn):
		self.activation_fn = fn

	def get_input_values(self):
		return self.get_values()


class FCLayer(Layer):
	def __init__(self, num_nodes, inp, activation_fn = ReLU):
		self.input_layer = inp
		self.layer_dim = num_nodes
		self.inp_dim = inp.get_dim()
		self.activation_fn = activation_fn
		self.initialize_values()

	def initialize_values(self):
		self.weights = np.random.normal(0, 1.0/np.sqrt(self.inp_dim),size=(self.inp_dim, self.layer_dim))
		self.biases = np.zeros((self.layer_dim,))
		self.values = self.activation_fn(np.dot(self.input_layer.get_values(), self.weights) + self.biases)

	def get_input(self):
		return self.input_layer

	def set_input(self, inp_layer):
		self.input_layer = inp_layer

	def get_weights(self):
		return self.weights

	def set_weights(self, matrix):
		if matrix.shape != self.weights.shape:
			raise ValueError("Weight matrices are not the same size")

		self.weights = matrix

	def get_biases(self):
		return self.biases

	def set_biases(self, matrix):
		if matrix.shape != self.biases.shape:
			raise ValueError("Weight matrices are not the same size")

		self.biases = matrix

	def get_input_values(self):
		return np.dot(self.input_layer.get_values(), self.weights) + self.biases

	def forward_prop(self):
		self.values = self.activation_fn(np.dot(self.input_layer.get_values(), self.weights) + self.biases)