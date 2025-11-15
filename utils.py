import numpy as np

def vectorize_tensor(T, B):
	"""
	Vectorizes the tensor T by selecting only the indices in B.

	Parameters:
	- T: numpy array of shape (num_samples, ...)
	- B: list of indices to select from each sample (array-like of shape (n_indices, n_dims))

	Returns:
	- V: vectorized representation of T of shape (num_samples, len(B))
	"""
	num_samples = T.shape[0]

	# Convert B to numpy array if it's not already
	B_array = np.asarray(B)

	# Handle edge case: if B is empty
	if len(B) == 0:
		return np.zeros((num_samples, 0))

	# Vectorized indexing: construct advanced indexing tuples
	# For each sample, we want T[i, b[0], b[1], ..., b[n_dims-1]] for all b in B
	# We can do this by creating indices for all samples at once
	sample_indices = np.arange(num_samples)[:, np.newaxis]  # Shape: (num_samples, 1)

	# B_array has shape (len(B), n_dims)
	# We need to create indexing arrays for each dimension
	if B_array.ndim == 1:
		# Special case: 1D indices
		V = T[sample_indices, B_array]
	else:
		# General case: multi-dimensional indices
		# Create indexing tuple: (sample_indices, B[:, 0], B[:, 1], ...)
		indexing_arrays = [sample_indices] + [B_array[:, dim] for dim in range(B_array.shape[1])]
		V = T[tuple(indexing_arrays)]

	return V

def reconstruct_tensor(V, T_shape, B):
	"""
	Reconstructs the tensor T from its vectorized representation V.

	Parameters:
	- V: vectorized representation of T of shape (num_samples, len(B))
	- T_shape: original shape of the tensor T (tuple)
	- B: list of indices that were used to create the vectorized representation (array-like of shape (n_indices, n_dims))

	Returns:
	- T: reconstructed tensor with the original shape, missing values filled with zero
	"""
	num_samples = V.shape[0]
	T = np.zeros(T_shape)

	# Convert B to numpy array if it's not already
	B_array = np.asarray(B)

	# Handle edge case: if B is empty
	if len(B) == 0:
		return T

	# Vectorized indexing for assignment
	sample_indices = np.arange(num_samples)[:, np.newaxis]  # Shape: (num_samples, 1)

	if B_array.ndim == 1:
		# Special case: 1D indices
		T[sample_indices, B_array] = V
	else:
		# General case: multi-dimensional indices
		# Create indexing tuple: (sample_indices, B[:, 0], B[:, 1], ...)
		indexing_arrays = [sample_indices] + [B_array[:, dim] for dim in range(B_array.shape[1])]
		T[tuple(indexing_arrays)] = V

	return T

def renormalize_image(image, threshold=False, threshold_value=10):
	"""
	Renormalize an image such that pixel values are between 0 and 255.
	Optionally, set small pixel values to 0.

	Parameters:
	- image: numpy array with shape (28, 28)
	- threshold: bool, if True, set small pixel values to 0
	- threshold_value: int, the value below which pixels are set to 0 (if threshold is True)

	Returns:
	- renormalized_image: numpy array with shape (28, 28), pixel values between 0 and 255
	"""
	# Ensure the image is a numpy array
	image = np.array(image)

	# Renormalize pixel values to be between 0 and 255
	min_val = np.min(image)
	max_val = np.max(image)
	renormalized_image = 255 * (image - min_val) / (max_val - min_val)

	# Apply threshold if needed
	if threshold:
		renormalized_image[renormalized_image < threshold_value] = 0

	return renormalized_image.astype(np.uint8)