import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function.

    Parameters:
    - x: ndarray
      Input values.

    Returns:
    - sigmoid: ndarray
      Sigmoid values.
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    """
    Logistic Regression classifier implementation for binary classification tasks.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the Logistic Regression classifier.

        Parameters:
        - lr: float, default=0.001
          Learning rate for gradient descent optimization.
        - n_iters: int, default=1000
          Number of iterations for training.

        Returns:
        None
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Logistic Regression model to the training data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input training data.
        - y: ndarray, shape (n_samples,)
          The target labels.

        Returns:
        None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data to be classified.

        Returns:
        - results: ndarray, shape (n_samples,)
          The predicted class labels for the input data.
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_pred)
        results = np.where(predictions <= 0.5, 0, 1)
        return results

        





    
        