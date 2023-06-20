import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - x1: ndarray
      The first point.
    - x2: ndarray
      The second point.

    Returns:
    - distance: float
      The Euclidean distance between x1 and x2.
    """
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    """
    K-Nearest Neighbors (KNN) classifier implementation for classification tasks.
    """

    def __init__(self, k=3):
        """
        Initialize the KNN classifier.

        Parameters:
        - k: int, default=3
          The number of nearest neighbors to consider.

        Returns:
        None
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the KNN model to the training data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input training data.
        - y: ndarray, shape (n_samples,)
          The target labels.

        Returns:
        None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data to be classified.

        Returns:
        - predictions: ndarray, shape (n_samples,)
          The predicted class labels for the input data.
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        """
        Perform the prediction for a single sample.

        Parameters:
        - x: ndarray, shape (n_features,)
          The input sample to be classified.

        Returns:
        - predicted_label: object
          The predicted class label for the input sample.
        """
        # Compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k closest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Determine the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Find the most common label
        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]

