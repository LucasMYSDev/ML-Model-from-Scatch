import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        x1 (numpy.ndarray): First point.
        x2 (numpy.ndarray): Second point.
        
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2)**2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        """
        K-means clustering algorithm.
        
        Args:
            K (int): Number of clusters. Default is 5.
            max_iters (int): Maximum number of iterations. Default is 100.
            plot_steps (bool): Whether to plot intermediate steps. Default is False.
        """
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        """
        Perform K-means clustering on the given data.
        
        Args:
            X (numpy.ndarray): Input data array.
        
        Returns:
            numpy.ndarray: Cluster labels for each data point.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        """
        Assign cluster labels to each data point based on the cluster they belong to.
        
        Args:
            clusters (list): List of clusters, where each cluster is a list of data point indices.
        
        Returns:
            numpy.ndarray: Cluster labels for each data point.
        """
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        """
        Create clusters by assigning data points to their closest centroids.
        
        Args:
            centroids (list): List of centroid points.
        
        Returns:
            list: List of clusters, where each cluster is a list of data point indices.
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        Find the index of the closest centroid to a given data point.
        
        Args:
            sample (numpy.ndarray): Data point.
            centroids (list): List of centroid points.
        
        Returns:
            int: Index of the closest centroid.
        """
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        """
        Calculate the new centroids based on the data points in each cluster.
        
        Args:
            clusters (list): List of clusters, where each cluster is a list of data point indices.
        
        Returns:
            numpy.ndarray: New centroid points.
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        """
        Check if the centroids have converged.
        
        Args:
            centroids_old (numpy.ndarray): Old centroid points.
            centroids (numpy.ndarray): New centroid points.
        
        Returns:
            bool: True if the centroids have converged, False otherwise.
        """
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return np.sum(distances) == 0

    def plot(self):
        """
        Plot the data points and centroids.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()



