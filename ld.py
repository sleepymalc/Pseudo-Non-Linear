# -*- coding: utf-8 -*-
"""CUDA-enabled LegendreDecomposition calculations"""
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from typing import Sequence
from scipy.special import logsumexp as scipy_logsumexp
# import itertools

try:
	import cupy as cp
	from cupyx.scipy.special import logsumexp as cupy_logsumexp
except ImportError:
	import numpy as cp
	from scipy.special import logsumexp as cupy_logsumexp
	def get_array_module(X):
		return np
	cp.get_array_module = get_array_module

def block_B(start_idx, end_idx):
	"""
	Create a block B of indexes for the center region.

	Parameters:
	- start_idx: tuple or list, starting indices for each dimension
	- end_idx: tuple or list, ending indices for each dimension

	Returns:
	- center_region_indexes: numpy array with shape (n, d), the indexes of the center region, where d is the number of dimensions
	"""
	if len(start_idx) != len(end_idx):
		raise ValueError("start_idx and end_idx must have the same number of dimensions.")

	# Generate all combinations of indices
	grids = [np.arange(start, end) for start, end in zip(start_idx, end_idx)]
	mesh_grids = np.meshgrid(*grids, indexing='ij')
	center_region_indexes = np.column_stack([grid.flatten() for grid in mesh_grids])

	return center_region_indexes

def step_B(shape, step_size):
	"""
	Create an index set for a tensor with a specified step size.

	Parameters:
	- shape: tuple, shape of the tensor (or matrix)
	- step_size: int or tuple of ints, step size along each dimension

	Returns:
	- index_set: numpy array with shape (n, len(shape)), the indices with the specified step size
	"""
	# Generate all combinations of indices with step size
	grids = [np.arange(0, size, step) for size, step in zip(shape, step_size)]
	mesh_grids = np.meshgrid(*grids, indexing='ij')
	index_set = np.column_stack([grid.flatten() for grid in mesh_grids])

	return index_set

def default_B(shape: Sequence[int], order: int, xp: ModuleType = np) -> NDArray[np.intp]:
	"""Vectorized implementation of the default B tensor.

	Args:
		shape: Shape of the corresponding X tensor.
		order: Order of the B tensor.
		xp (ModuleType): Array module, either numpy (CPU) or cupy (CUDA/GPU)

	Returns:
		array-like: Default B tensor of specified order.
	"""
	B = xp.indices(shape).reshape(len(shape), -1).T
	mask = (B != 0).sum(axis=1) <= order
	return B[mask]

def kNN(input_theta, training_theta, k=1, eps: float = 1.0e-5, metric='kl'):
	"""Vectorized k-nearest neighbors computation.

	Args:
		input_theta: Query point theta coordinates
		training_theta: Array of training theta coordinates (n_samples, ...)
		k: Number of nearest neighbors
		eps: Small constant for numerical stability
		metric: Distance metric ('kl' or 'euclidean')

	Returns:
		Indices of k nearest neighbors
	"""
	xp = cp.get_array_module(input_theta)

	if xp == cp:
		logsumexp = cupy_logsumexp
	else:
		logsumexp = scipy_logsumexp

	# Convert training_theta to array if needed
	if isinstance(training_theta, list):
		training_theta = xp.array(training_theta)

	if metric == 'euclidean':
		# Vectorized euclidean distance
		# Flatten tensors for distance computation
		# training_theta has shape (n_samples, *tensor_shape), flatten to (n_samples, -1)
		# input_theta has shape (*tensor_shape,), flatten to (-1,)
		training_theta_flat = training_theta.reshape(training_theta.shape[0], -1)
		input_theta_flat = input_theta.reshape(-1)

		# Compute distances: shape (n_samples,)
		distances = xp.linalg.norm(training_theta_flat - input_theta_flat[None, :], axis=1)

	elif metric == 'kl':
		# For KL divergence, we need to compute Q for each sample
		# This is still faster than the loop as we batch the operations
		n_samples = training_theta.shape[0]
		distances = xp.zeros(n_samples, dtype=xp.float64)

		# Compute Q for input once
		input_Q = get_Q(input_theta, logsumexp, eps=eps, xp=xp)

		# Batch compute distances (still need loop but operations are vectorized)
		for i in range(n_samples):
			training_Q = get_Q(training_theta[i], logsumexp, eps=eps, xp=xp)
			distances[i] = kl(input_Q, training_Q, xp)

	else:
		raise ValueError(f"Unknown metric: {metric}")

	# Use argpartition for O(n) complexity instead of O(n log n) sort
	# This is much faster when k << n
	if k >= len(distances):
		k_nearest_indices = xp.arange(len(distances))
	else:
		k_nearest_indices = xp.argpartition(distances, k-1)[:k]
		# Sort only the k nearest
		k_nearest_indices = k_nearest_indices[xp.argsort(distances[k_nearest_indices])]

	return k_nearest_indices

def kl(P: NDArray[np.float64], Q: NDArray[np.float64], xp: ModuleType = cp) -> np.float64:
	"""Kullback-Leibler divergence.

	Args:
		P: P tensor
		Q: Q tensor
		xp (ModuleType): Array module, either numpy (CPU) or cupy

	Returns:
		KL divergence.
	"""
	return xp.sum(P * xp.log(P / Q)) - xp.sum(P) + xp.sum(Q)


def get_eta(Q: NDArray[np.float64], D: int, xp: ModuleType = cp) -> NDArray[np.float64]:
	"""Eta tensor.

	Args:
		Q: Q tensor
		D: Dimensionality
		xp (ModuleType): Array module, either numpy (CPU) or cupy

	Returns:
		Eta tensor.
	"""
	for i in range(D):
		Q = xp.flip(xp.cumsum(xp.flip(Q, axis=i), axis=i), axis=i)
	return Q


def get_h(theta: NDArray[np.float64], D: int, xp: ModuleType = cp) -> NDArray[np.float64]:
	"""H tensor.

	Args:
		theta: Theta tensor
		D: Dimensionality
		xp (ModuleType): Array module, either numpy (CPU) or cupy

	Returns:
		Updated theta.
	"""
	for i in range(D):
		theta = xp.cumsum(theta, axis=i)
	return theta

def get_Q(theta: NDArray[np.float64], logsumexp, eps, xp: ModuleType = cp) -> NDArray[np.float64]:
	"""Compute the probability tensor Q from parameter theta.

	Args:
		theta : Theta tensor
		logsumexp : function
		eps : (see paper)
		xp (ModuleType): Array module, either numpy (CPU) or cupy

	Returns:
	Q : probability tensor of theta
	"""
	# theta = xp.asarray(theta)
	# eps = xp.asarray(eps)

	# theta => H
	D = len(theta.shape)
	Hq = get_h(theta, D, xp)

	# H => Q
	logQ_ = Hq
	logQ = logQ_ - logsumexp(logQ_)
	Q = xp.exp(logQ) + eps
	Q /= Q.sum()

	return Q

def compute_G_matvec(v, eta, uuu, vvv, I_flat, J_flat, K_flat, xp):
	"""Compute matrix-vector product G @ v without storing G.

	This is a memory-efficient way to compute the Fisher information matrix
	multiplication, which is the bottleneck for large-scale problems.

	Args:
		v: Vector to multiply with G
		eta: Eta tensor
		uuu, vvv: Lower triangular indices
		I_flat, J_flat, K_flat: Precomputed flat indices
		xp: Array module (numpy or cupy)

	Returns:
		G @ v
	"""
	result = xp.zeros_like(v)

	# Compute G entries on-the-fly
	# G[i, j] = eta[K] - eta[I] * eta[J]
	# where K = max(B[i], B[j]), I = B[i], J = B[j]

	G_values = xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat)

	# Build result using the symmetric structure - vectorized approach
	# Since G is symmetric, we only computed lower triangular part
	uuu_array = xp.asarray(uuu) if not isinstance(uuu, xp.ndarray) else uuu
	vvv_array = xp.asarray(vvv) if not isinstance(vvv, xp.ndarray) else vvv

	# Accumulate contributions from lower triangle
	xp.add.at(result, uuu_array, G_values * v[vvv_array])
	# Accumulate contributions from upper triangle (symmetric part)
	mask = uuu_array != vvv_array  # Only off-diagonal elements
	xp.add.at(result, vvv_array[mask], G_values[mask] * v[uuu_array[mask]])

	return result

def LD(X: NDArray[np.float64],
	B: NDArray[np.intp] | list[tuple[int, ...]] | None = None,
	order: int = 2,
	n_iter: int = 10,
	lr: float = 1.0,
	eps: float = 1.0e-5,
	error_tol: float = 1.0e-5,
	ngd: bool = True,
	verbose: bool = True,
	gpu: bool = True,
	exit_abs: bool = True,
	dtype: np.dtype | None = None,
	iterative_solver: bool = False,
	solver_tol: float = 1.0e-6,
	solver_maxiter: int = 100,
) -> tuple[list[list[float]], np.float64, NDArray[np.float64], NDArray[np.float64]]:
	"""Compute many-body tensor approximation.

	Args:
		X: Input tensor.
		B: B tensor.
		order: Order of default tensor B, if not provided.
		n_iter: Maximum number of iteration.
		lr: Learning rate.
		eps: (see paper).
		error_tol: KL divergence tolerance for the iteration.
		ngd: Use natural gradient.
		verbose: Print debug messages.
		gpu: Use GPU (CUDA or ROCm depending on the installed CuPy version).
		exit_abs: Previous implementation (wrongly?) uses kl- kl_prev as iteration exit criterion.
			Use abs(kl - kl_prev) instead.
		dtype: By default, the data-type is inferred from the input data.
		iterative_solver: Use iterative solver (Conjugate Gradient) instead of direct solver.
			This saves memory for large B but may be slower for small problems.
		solver_tol: Tolerance for iterative solver convergence.
		solver_maxiter: Maximum iterations for iterative solver.

	Returns:
		history_kl: KL divergence history.
		history_norm: norm difference history.
		scaleX: Scaled X tensor.
		Q: Q tensor.
		theta: Theta.
	"""
	if exit_abs:
		def within_tolerance(kld: np.float64, prev_kld: np.float64):
			return abs(prev_kld - kld) < error_tol
	else:
		def within_tolerance(kld: np.float64, prev_kld: np.float64):
			return prev_kld - kld < error_tol

	if gpu:
		X = cp.asarray(X, dtype=dtype)
		eps = cp.asarray(eps, dtype=dtype)
		lr = cp.asarray(lr, dtype=dtype)
		logsumexp = cupy_logsumexp
	else:
		logsumexp = scipy_logsumexp

	xp = cp.get_array_module(X)
	D = len(X.shape)
	S = X.shape

	if verbose:
		print("Constructing B")
	if B is None:
		B = default_B(S, order, xp)

	B_array = xp.array(B)
	B_flat = xp.ravel_multi_index(B_array.T, S)  # type: ignore
	if verbose:
		print("B shape:", B_flat.shape)

	scaleX = xp.sum(X + eps)
	P = (X + eps) / scaleX

	Q = xp.ones(P.shape, dtype=dtype)  # TODO: ones_like?
	Q = Q / xp.sum(Q)

	### eta
	eta_b = xp.empty((len(B),), dtype=dtype)
	theta_b = xp.zeros((len(B),), dtype=dtype)

	### eta_hat
	eta_hat = get_eta(P, D, xp)
	eta_hat_b = xp.take(eta_hat, B_flat)

	# Precompute indices for G matrix computation
	uuu, vvv = xp.tril_indices(len(B), 0)
	I_flat = B_flat[uuu]
	J_flat = B_flat[vvv]
	K_flat = xp.ravel_multi_index(xp.maximum(B_array[uuu], B_array[vvv]).T, S)  # type: ignore

	# Only allocate G if using direct solver
	if not iterative_solver:
		G = xp.zeros((len(B), len(B)), dtype=dtype)
		uv = xp.ravel_multi_index(xp.stack((uuu, vvv)), (len(B), len(B)))  # type: ignore

	# evaluation
	history_kl = []
	kld = kl(P, Q, xp)
	history_kl.append(float(kld))
	prev_kld = np.inf

	history_norm = []
	norm = np.inf
	history_norm.append(norm)

	early_stop = False

	if verbose:
		print("iter=", 0, "kl=", kld, "mse=", xp.mean((P - Q) ** 2), "eta_difference_norm=", norm)

	for i in range(n_iter):
		# compute eta
		eta = get_eta(Q, D, xp)
		eta_b = xp.take(eta, B_flat)

		# update theta_b
		if ngd:
			rhs = lr * (eta_b[1:] - eta_hat_b[1:])

			if iterative_solver:
				# Use iterative solver (memory-efficient)
				# Define matrix-vector product function
				def matvec(v_extended):
					# Extend v to full dimension (add back the first element as 0)
					v_full = xp.zeros(len(B), dtype=dtype)
					v_full[1:] = v_extended
					result_full = compute_G_matvec(v_full, eta, uuu, vvv, I_flat, J_flat, K_flat, xp)
					return result_full[1:]

				# Use conjugate gradient - note: need to implement or use scipy
				if xp == cp:
					# CuPy has cupyx.scipy.sparse.linalg.cg
					try:
						from cupyx.scipy.sparse.linalg import cg
						v, info = cg(matvec, rhs, tol=solver_tol, maxiter=solver_maxiter, atol=0)
						if info != 0 and verbose:
							print(f"Warning: CG did not converge, info={info}")
					except ImportError:
						# Fallback to direct solve if CG not available
						if verbose:
							print("Warning: cupyx.scipy.sparse.linalg.cg not available, using direct solver")
						iterative_solver = False
				else:
					# NumPy: use scipy.sparse.linalg.cg
					from scipy.sparse.linalg import LinearOperator, cg
					n = len(B) - 1
					A_op = LinearOperator((n, n), matvec=matvec, dtype=dtype)
					v, info = cg(A_op, rhs, tol=solver_tol, maxiter=solver_maxiter, atol=0)
					if info != 0 and verbose:
						print(f"Warning: CG did not converge, info={info}")

				if not iterative_solver:
					# Fallback happened, use direct solver
					xp.put(G, uv, xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat))
					GG = G + G.T - xp.diag(G.diagonal())
					v = xp.linalg.solve(GG[1:, 1:], rhs)
			else:
				# Use direct solver (original implementation)
				xp.put(G, uv, xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat))
				GG = G + G.T - xp.diag(G.diagonal())
				v = xp.linalg.solve(GG[1:, 1:], rhs)

			theta_b[1:] -= v
		else:
			theta_b[1:] -= lr * (eta_b[1:] - eta_hat_b[1:])

		# theta_b=>theta
		theta = xp.zeros(S, dtype=dtype)
		xp.put(theta, B_flat, theta_b)

		# theta => Q
		Q = get_Q(theta, logsumexp, eps=eps, xp=xp)
		Q = Q / xp.sum(Q)

		# evaluation
		norm = xp.linalg.norm(eta_b - eta_hat_b)
		history_norm.append(float(norm))

		kld = kl(P, Q, xp)
		history_kl.append(float(kld))
		if verbose:
			print("iter=", i + 1, "kl=", kld, "mse=", xp.mean((P - Q) ** 2), "eta_difference_norm=", norm)

		if norm < error_tol or within_tolerance(kld, prev_kld):
			early_stop = True
			break

		prev_kld = kld

		# lower the learning rate
		if i in [200, 500, 900]:
			lr *= 0.1

	if not early_stop:
		print("Warning: Not Converged. Consider increasing the number of iterations.")

	if gpu:
		scaleX = scaleX.get()  # type: ignore
		Q = Q.get()  # type: ignore
		theta = theta.get()  # type: ignore

	return history_kl, history_norm, scaleX, Q, theta  # type: ignore

def BP(
	theta: NDArray[np.float64],
	X: NDArray[np.float64],
	submfd_eta: NDArray[np.float64],
	scale: np.float64,
	B: NDArray[np.intp] | list[tuple[int, ...]] | None = None,
	order: int = 2,
	n_iter: int = 10,
	lr: float = 1.0,
	eps: float = 1.0e-5,
	error_tol: float = 1.0e-4,
	ngd: bool = True,
	verbose: bool = True,
	gpu: bool = True,
	exit_abs: bool = True,
	exit_mode: str = 'kl',
	dtype: np.dtype | None = None,
	iterative_solver: bool = False,
	solver_tol: float = 1.0e-6,
	solver_maxiter: int = 100,
) -> tuple[list[list[float]], np.float64, NDArray[np.float64], NDArray[np.float64]]:
	"""Compute many-body tensor approximation (backward projection).

	Args:
		theta: theta coordinates
		X: submanifold's original tensor(s)
		submfd_eta: The m-flat submanifold specified by eta(s).
		scale: The scale of the pre-projection of P.
		B: B tensor.
		order: Order of default tensor B, if not provided.
		n_iter: Maximum number of iteration.
		lr: Learning rate.
		eps: (see paper).
		error_tol: KL divergence tolerance for the iteration.
		ngd: Use natural gradient.
		verbose: Print debug messages.
		gpu: Use GPU (CUDA or ROCm depending on the installed CuPy version).
		exit_abs: Previous implementation (wrongly?) uses kl- kl_prev as iteration exit criterion.
			Use abs(kl - kl_prev) instead.
		dtype: By default, the data-type is inferred from the input data.
		iterative_solver: Use iterative solver (Conjugate Gradient) instead of direct solver.
		solver_tol: Tolerance for iterative solver convergence.
		solver_maxiter: Maximum iterations for iterative solver.

	Returns:
		history_kl: KL divergence history.
		history_norm: norm difference history.
		P: P tensor.
		theta: Theta.
	"""
	if exit_abs:
		def within_tolerance(kld: np.float64, prev_kld: np.float64):
			return abs(prev_kld - kld) < error_tol
	else:
		def within_tolerance(kld: np.float64, prev_kld: np.float64):
			return prev_kld - kld < error_tol

	if gpu:
		theta = cp.asarray(theta, dtype=dtype)
		X = cp.asarray(X, dtype=dtype)
		submfd_eta = cp.asarray(submfd_eta, dtype=dtype)
		scale = cp.asarray(scale, dtype=dtype)
		eps = cp.asarray(eps, dtype=dtype)
		lr = cp.asarray(lr, dtype=dtype)
		logsumexp = cupy_logsumexp
	else:
		logsumexp = scipy_logsumexp
	k = X.shape[0]
	lr /= k

	xp = cp.get_array_module(theta)

	P_ = get_Q(theta, logsumexp, eps=eps, xp=xp)
	D = len(P_.shape)
	S = P_.shape

	P = (P_ + eps) / xp.sum(P_ + eps)

	if verbose:
		print("Constructing B")
	if B is None:
		B = default_B(S, order, xp)

	B_array = xp.array(B)
	B_flat = xp.ravel_multi_index(B_array.T, S)  # type: ignore
	if verbose:
		print("B shape:", B_flat.shape)

	full_B = default_B(S, D, xp)
	full_B_array = xp.array(full_B)
	full_B_flat = xp.ravel_multi_index(full_B_array.T, S)  # type: ignore

	theta_full_b = xp.take(theta, full_B_flat)

	### eta_hat (the target)
	eta_hat = submfd_eta
	eta_hat_b = xp.take(eta_hat, B_flat)
	eta_hat_full_b = xp.take(eta_hat, full_B_flat)

	# Precompute indices for G matrix computation
	uuu, vvv = xp.tril_indices(len(full_B), 0)
	I_flat = full_B_flat[uuu]
	J_flat = full_B_flat[vvv]
	K_flat = xp.ravel_multi_index(xp.maximum(full_B_array[uuu], full_B_array[vvv]).T, S)  # type: ignore

	# Only allocate G if using direct solver
	if not iterative_solver:
		G = xp.zeros((len(full_B), len(full_B)), dtype=dtype)
		uv = xp.ravel_multi_index(xp.stack((uuu, vvv)), (len(full_B), len(full_B)))  # type: ignore

	# evaluation
	history_kl = []
	kld = 0
	for x in X:
		kld += kl(P, X, xp)
	history_kl.append(float(kld))
	prev_kld = np.inf

	history_norm = []
	norm = np.inf
	history_norm.append(norm)

	early_stop = False

	if verbose:
		print("iter=", 0, "kl=", kld, "eta_difference_norm=", norm)

	for i in range(n_iter):
		# compute eta
		eta = get_eta(P, D, xp)
		eta_b = xp.take(eta, B_flat)
		eta_full_b = xp.take(eta, full_B_flat)

		# update theta_b
		if ngd:
			rhs = lr * (eta_full_b[1:] - eta_hat_full_b[1:])

			if iterative_solver:
				# Use iterative solver (memory-efficient)
				def matvec(v_extended):
					v_full = xp.zeros(len(full_B), dtype=dtype)
					v_full[1:] = v_extended
					result_full = compute_G_matvec(v_full, eta, uuu, vvv, I_flat, J_flat, K_flat, xp)
					return result_full[1:]

				if xp == cp:
					try:
						from cupyx.scipy.sparse.linalg import cg
						v, info = cg(matvec, rhs, tol=solver_tol, maxiter=solver_maxiter, atol=0)
						if info != 0 and verbose:
							print(f"Warning: CG did not converge, info={info}")
					except ImportError:
						if verbose:
							print("Warning: cupyx.scipy.sparse.linalg.cg not available, using direct solver")
						iterative_solver = False
				else:
					from scipy.sparse.linalg import LinearOperator, cg
					n = len(full_B) - 1
					A_op = LinearOperator((n, n), matvec=matvec, dtype=dtype)
					v, info = cg(A_op, rhs, tol=solver_tol, maxiter=solver_maxiter, atol=0)
					if info != 0 and verbose:
						print(f"Warning: CG did not converge, info={info}")

				if not iterative_solver:
					# Fallback to direct solver
					xp.put(G, uv, xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat))
					GG = G + G.T - xp.diag(G.diagonal())
					v = xp.linalg.solve(GG[1:, 1:], rhs)
			else:
				# Use direct solver
				xp.put(G, uv, xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat))
				GG = G + G.T - xp.diag(G.diagonal())
				v = xp.linalg.solve(GG[1:, 1:], rhs)

			theta_full_b[1:] -= v
		else:
			theta_full_b[1:] -= lr * (eta_full_b[1:] - eta_hat_full_b[1:])

		# theta_b=>theta
		theta = xp.zeros(S, dtype=dtype)
		xp.put(theta, full_B_flat, theta_full_b)

		# theta => P
		P = get_Q(theta, logsumexp, eps=eps, xp=xp)
		P = P / xp.sum(P)

		# evaluation
		if exit_mode == 'kl':
			kld = 0
			for x in X:
				kld += kl(P, x, xp)
			history_kl.append(float(kld))
			if within_tolerance(kld, prev_kld):
				early_stop = True
				break
			if verbose:
				print("iter=", i + 1, "kl=", kld, "kl=", kld)

			prev_kld = kld
		elif exit_mode == 'mse':
			norm = xp.linalg.norm(eta_b - eta_hat_b)
			history_norm.append(float(norm))
			if norm < error_tol:
				early_stop = True
				break
			if verbose:
				print("iter=", i + 1, "mse=", kld, "eta_difference_norm=", norm)


		# lower the learning rate
		if i in [200, 500, 900]:
			lr *= 0.1

	if not early_stop:
		print("Warning: Not Converged. Consider increasing the number of iterations.")
	else:
		print("Converged.")

	P = P * scale

	if gpu:
		P = P.get()  # type: ignore
		theta = theta.get()  # type: ignore

	return history_kl, history_norm, P, theta  # type: ignore