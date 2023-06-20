import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation for dimensionality reduction.
    """

    def __init__(self, n_components):
        """
        Initialize the PCA model.

        Parameters:
        - n_components: int
          The number of principal components to keep.

        Returns:
        None
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the input data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data.

        Returns:
        None
        """
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transform the input data using the learned PCA model.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data to be transformed.

        Returns:
        - X_transformed: ndarray, shape (n_samples, n_components)
          The transformed data.
        """
        X = X - self.mean
        return np.dot(X, self.components)





