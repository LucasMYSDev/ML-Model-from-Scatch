import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier implementation using stochastic gradient descent.

    Parameters:
    - learning_rate (float): The learning rate for the gradient descent update (default=0.001).
    - lambda_param (float): The regularization parameter for controlling the trade-off between the margin and training error (default=0.01).
    - n_iters (int): The number of iterations for training (default=1000).

    Attributes:
    - lr (float): The learning rate for the gradient descent update.
    - lambda_param (float): The regularization parameter.
    - n_iters (int): The number of training iterations.
    - w (ndarray): The weight vector for the SVM decision function.
    - b (float): The bias term for the SVM decision function.

    Methods:
    - fit(X, y): Fit the SVM model to the training data.
    - predict(X): Predict the class labels for new data.

    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM classifier.

        Args:
        - learning_rate (float): The learning rate for the gradient descent update (default=0.001).
        - lambda_param (float): The regularization parameter (default=0.01).
        - n_iters (int): The number of training iterations (default=1000).
        """

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.

        Args:
        - X (ndarray): The input feature matrix of shape (n_samples, n_features).
        - y (ndarray): The target labels of shape (n_samples,).

        """

        n_samples, n_features = X.shape

        # Transform class labels to +1 and -1
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict the class labels for new data.

        Args:
        - X (ndarray): The input feature matrix of shape (n_samples, n_features).

        Returns:
        - ndarray: The predicted class labels of shape (n_samples,).

        """

        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
