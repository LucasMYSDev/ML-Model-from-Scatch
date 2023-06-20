from collections import Counter
import numpy as np
from DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        """
        Initialize the RandomForest object.

        Args:
        - n_trees: Number of decision trees in the random forest (default: 10)
        - max_depth: Maximum depth of each decision tree (default: 10)
        - min_samples_split: Minimum number of samples required to split a node in each decision tree (default: 2)
        - n_features: Number of features to consider for each split in each decision tree (default: None, which uses all features)
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Build the random forest by fitting decision trees to bootstrap samples of the training data.

        Args:
        - X: Input features of shape (n_samples, n_features)
        - y: Target labels of shape (n_samples,)
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Create a new decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            
            # Generate bootstrap samples from the training data
            X_sample, y_sample = self._bootstrap_samples(X, y)
            
            # Fit the decision tree to the bootstrap samples
            tree.fit(X_sample, y_sample)
            
            # Add the trained decision tree to the random forest
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        """
        Generate bootstrap samples from the training data.

        Args:
        - X: Input features of shape (n_samples, n_features)
        - y: Target labels of shape (n_samples,)

        Returns:
        - X_sample: Bootstrap samples of X
        - y_sample: Bootstrap samples of y
        """
        n_samples = X.shape[0]
        
        # Generate random indices with replacement
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Create bootstrap samples using the random indices
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """
        Find the most common label in a given array.

        Args:
        - y: Array of labels

        Returns:
        - most_common: Most common label
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Make predictions for new input data.

        Args:
        - X: Input features of shape (n_samples, n_features)

        Returns:
        - predictions: Array of predicted labels
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose predictions to align them by samples instead of trees
        tree_preds = np.swapaxes(predictions, 0, 1)
        
        # Compute the most common label for each sample
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions





        