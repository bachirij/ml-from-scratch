import numpy as np

# DBSCAN class
class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps=eps
        self.min_samples=min_samples


    def fit_predict(self, X):
        """
        Compute labels for each data point.
        """
        labels = self._label_points(X)
        return labels 
    

    def _get_neighbors(self, X, point_idx):
        """
        Get neighbors of point of index point_idx within eps distance.
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return list(np.where(distances <= self.eps)[0])
    

    def _expand_cluster(self, X, labels, neighbors, cluster_id):
        """
        Expand cluster from point_idx.
        """
        queue = list(neighbors)
        while queue:
            neighbor_idx = queue.pop(0)
            if labels[neighbor_idx] != -1:  # Already classified as noise or part of another cluster
                continue
            else: # Unvisited
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)


    def _label_points(self, X):
        """
        Compute abel points based on DBSCAN criteria.
        """
        # Initialize all points as noise (label -1)
        labels = np.full(X.shape[0], -1)  
        cluster_id = 0

        # Iterate through each point in the dataset
        for point_idx in range(X.shape[0]):
            if labels[point_idx] != -1:
                continue  # Skip already labeled points

            neighbors = self._get_neighbors(X, point_idx)

            # If number of neighbors is less than min_samples, mark as noise
            if len(neighbors) >= self.min_samples:
                labels[point_idx] = cluster_id  # Assign cluster ID
                self._expand_cluster(X, labels, neighbors, cluster_id)
                cluster_id += 1

        return labels
    
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

    print(f"From scratch model:")
    model = DBSCAN(eps=0.2, min_samples=5)
    labels = model.fit_predict(X)
    print("Scratch clusters:", np.unique(labels))

    print(f"Scikit-Learn model:")
    from sklearn.cluster import DBSCAN
    sk_model = DBSCAN(eps=0.2, min_samples=5)
    sk_labels = sk_model.fit_predict(X)
    print("Sklearn clusters:", np.unique(sk_labels))