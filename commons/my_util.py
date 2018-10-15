import numpy as np

def shuffle_dataset(x, t):
	"""
	Shuffle the data set

	Parameters
	----------
	x : training data
	t : test data

	Returns
	----------
	"""

	p = np.random.permutation(x.shape[0])
	x = x[p,:] if x.ndim == 2 else x[p,:,:,:]
	t = [p]

	return x, t

