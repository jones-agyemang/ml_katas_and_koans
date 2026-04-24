import numpy as np

"""
Goal: Master array manipulation without loops
"""

"""
# Challenge

Constraints:
  - No use of "for" loops
"""
# Given:
np.random.seed(42)
X = np.random.randn(100, 5)

# 1. Standardise each column (mean=0, std=1)
means = X.mean(axis=0)
std = X.std(axis=0)

X_standardised = (X - means) / std

assertion_msg = "Expected shape (100, 5), got: {X_standardised.shape}"
assert X_standardised.shape == (100, 5), assertion_msg

# 2. Compute Covariance matrix manually (no `np.cov`)
n_samples = X.shape[0]
cov_matrix = (X_standardised.T @ X_standardised) / (n_samples - 1)

assertion_msg = "Expected shape (5, 5), got: {cov_matrix.shape}"
print(f"Shape of covariance matrix: {cov_matrix.shape}")

# 3. Extract eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"Eigenvectors: {eigenvectors}")
print(f"Eigenvalues: {eigenvalues}")
