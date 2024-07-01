import numpy as np


def get_numeric_grad(f, x, eps):
	"""
	Function to calculate numeric gradient of f function in x.

	Parameters
	----------
	f : callable
	x : numpy.ndarray
		1d array, function argument
	eps : float
		Tolerance

	Returns
	-------
	: numpy.ndarray
		Numeric gradient.
	"""
	numeric_gradient = np.zeros(x.shape)
	f_x = f(x)

	for i in range(x.shape[0]):
		eps_vector = np.zeros(x.shape[0])
		eps_vector[i] = eps
		numeric_gradient[i] = (f(x + eps_vector) - f_x) / eps
	return numeric_gradient