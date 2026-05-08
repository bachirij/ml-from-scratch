import numpy as np

# KMeans class
class KMeans:
    def __init__(self, K=3, n_iters=100, n_init=10, random_state=None):
        self.K=K
        self.n_iters=n_iters
        self.n_init=n_init
        self.random_state=random_state
        self.centroids=None

    def fit(self, X):
        """
        Compute k-means clustering.
        """
        # Initialize best inertia to infinity
        best_inertia = np.inf   

        # Compute k-means algorithm n_init times, and keep the result with best inertia
        for _ in range (self.n_init):

            # Initialize centroids randomly from the dataset
            centroids = self._initialize_centroids(X)

            # k-means iterations
            for i in range(self.n_iters):
                # Step 1: Assign clusters
                labels = self._assign_clusters(X, centroids)

                # Step 2: Update centroids
                new_centroids = self._update_centroids(X, labels)

                # Check for convergence (stops if centroids do not change)
                if np.allclose(centroids, new_centroids):
                    print(f"Converged after {i} iterations.")
                    break
                
                centroids = new_centroids

            # Compute inertia
            inertia = self._compute_inertia(X, labels, centroids)

            # Compare inertias
            if inertia < best_inertia:
                best_inertia = inertia

                # If best run, update centroids, labels, inertia
                self.centroids = centroids
                self.labels_ = labels
                self.inertia_ = inertia


    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        if self.centroids is None:
            raise Exception("Model has not been fitted yet.")
        return self._assign_clusters(X, self.centroids)

    def _initialize_centroids(self, X):
        """
        Initialize K random centroids within X.
        """
        n_samples = X.shape[0]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(n_samples, size=self.K, replace=False) # K random indices among n_samples
        centroids = X[indices]
        return centroids


    def _assign_clusters(self, X, centroids):
        """
        Assign and update data points and centroids within clusters.
        """
        n_samples = X.shape[0]
        K = centroids.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Compute the euclidian distance between X[i] and each centroids
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            # Get the index of the closest centroid
            closest_centroid = np.argmin(distances)
            # Assign the label of the closest centroid to labels[i]
            labels[i] = closest_centroid
        
        return labels


    def _update_centroids(self, X, labels):
        """
        Update centroids by calculating the mean of the points assigned to each cluster.
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.K, n_features))
        
        for k in range(self.K):
            # Get the points assigned to cluster k
            cluster_points = X[labels == k]
            # Update the centroid of cluster k as the mean of its points
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
        
        return centroids


    def _compute_inertia(self, X, labels, centroids):
        """
        Computes the Within-Cluster Sum of Squares (WCSS).
        """
        return np.sum(np.linalg.norm(X - centroids[labels], axis=1) ** 2)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    n_samples = 300
    X, y_true = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8, random_state=42)
    
    print(f"From scratch model:")
    model = KMeans(K=3, n_init=10, random_state=42)
    model.fit(X)
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Centroids:\n{model.centroids}")

    from sklearn.metrics import silhouette_score
    print(f"Silhouette score: {silhouette_score(X, model.labels_):.3f}")

    print(f"Scikit-Learn model:")
    from sklearn.cluster import KMeans as SKLearnKMeans
    sklearn_model = SKLearnKMeans(n_clusters=3, n_init=10, random_state=42)
    sklearn_model.fit(X)
    print(f"Inertia: {sklearn_model.inertia_:.2f}")
    print(f"Centroids:\n{sklearn_model. cluster_centers_}")
    print(f"Silhouette score: {silhouette_score(X, sklearn_model.labels_):.3f}")