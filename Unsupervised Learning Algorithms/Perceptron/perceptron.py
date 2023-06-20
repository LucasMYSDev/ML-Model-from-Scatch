import numpy as np

def unit_step_func(x):
    """
    Apply the unit step function to the input.

    Parameters:
    - x: ndarray
      The input values.

    Returns:
    - ndarray
      The output values after applying the unit step function.
    """
    return np.where(x > 0, 1, 0)


class Perceptron:
    """
    Perceptron classifier implementation for binary classification tasks.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initialize the Perceptron classifier.

        Parameters:
        - learning_rate: float, default=0.01
          The learning rate for updating the weights.
        - n_iters: int, default=1000
          The number of iterations for training.

        Returns:
        None
        """
        self.lr = learning_rate
        self.iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Perceptron model to the training data.

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
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data to be classified.

        Returns:
        - y_predicted: ndarray, shape (n_samples,)
          The predicted class labels for the input data.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    

    # Testing

if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=4, cluster_std=2, random_state=2123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()


    



