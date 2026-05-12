import numpy as np

# PCA class
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Compute mean of features
        X_mean = np.mean(X, axis=0) 

        # Center features
        X_centered = X - X_mean

        # Compute covariance matrix
        C = (1/(X.shape[0]-1)) * X_centered.T @ X_centered

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Keep first n_components vectors
        W = eigenvectors[:, :self.n_components]

        # Attributes assignment
        self.components_ = W
        self.mean_ = X_mean
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()


    def transform(self, X): 
        # X projection into the new subspace
        Z = (X - self.mean_) @ self.components_
        return Z


    def fit_transform(self, X):
        self.fit(X)
        Z = self.transform(X)
        return Z

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    pca = PCA(n_components=1)
    Z = pca.fit_transform(X)
    print("Explained Variance Ratio =", pca.explained_variance_ratio_)
    print("Components =", pca.components_.shape)
    print("Projection matrix shape:", Z.shape)


