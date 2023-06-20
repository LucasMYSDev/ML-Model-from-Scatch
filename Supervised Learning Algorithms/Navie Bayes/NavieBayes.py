import numpy as np

class NaiveBayes:
    """
    A Naive Bayes classifier implementation for multi-class classification tasks.
    """

    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input training data.
        - y: ndarray, shape (n_samples,)
          The target labels.

        Returns:
        None
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features)
          The input data to be classified.

        Returns:
        - y_pred: ndarray, shape (n_samples,)
          The predicted class labels for the input data.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Perform the prediction for a single sample.

        Parameters:
        - x: ndarray, shape (n_features,)
          The input sample to be classified.

        Returns:
        - predicted_class: object
          The predicted class label for the input sample.
        """
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior += prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Compute the probability density function (PDF) for a feature given a class.

        Parameters:
        - class_idx: int
          The index of the class.
        - x: ndarray, shape (n_features,)
          The input feature vector.

        Returns:
        - pdf: ndarray, shape (n_features,)
          The computed PDF for each feature given the class.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator





    

