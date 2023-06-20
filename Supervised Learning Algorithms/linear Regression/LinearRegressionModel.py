import numpy as np

class LinearRegression:
    """
    Linear Regression model implementation for predicting continuous numeric values.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the Linear Regression model.

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
        Fit the Linear Regression model to the training data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input training data.
        - y: ndarray, shape (n_samples,)
          The target values.

        Returns:
        None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict the target values for the input data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data to be predicted.

        Returns:
        - y_pred: ndarray, shape (n_samples,)
          The predicted target values for the input data.
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


