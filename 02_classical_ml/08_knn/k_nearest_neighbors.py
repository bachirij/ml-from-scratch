import numpy as np

# KNNClassifier Class
class KNNClassifier():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        y_pred = [self._predict_one(x) for x in X_test]
        return np.array(y_pred)
    
    def _euclidian_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2) # L2 norm, same as: np.sqrt(np.sum((p1 - p2)**2))
    
    def _predict_one(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.array([self._euclidian_distance(x, x_train) for x_train in self.X_train])
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common class label among the neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    

# KNNRegressor Class
class KNNRegressor():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        y_pred = [self._predict_one(x) for x in X_test]
        return np.array(y_pred)
    
    def _euclidian_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2) # L2 norm, same as: np.sqrt(np.sum((p1 - p2)**2))
    
    def _predict_one(self, x):
        # Compute distances between x and all examples in the training set
        distances = np.array([self._euclidian_distance(x, x_train) for x_train in self.X_train])
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor
        k_nearest_labels = self.y_train[k_indices]
        # Return the average of the k nearest neighbor labels
        return np.mean(k_nearest_labels)
    

if __name__ == "__main__":
    # Example usage
    print("Testing KNNClassifier...")
    from sklearn.metrics import accuracy_score
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1]) # OR logic gate
    model = KNNClassifier(k=3)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("Accuracy:", accuracy_score(y, predictions))

    print("\nTesting KNNRegressor...")
    from sklearn.metrics import mean_squared_error
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5]) # y = x
    model = KNNRegressor(k=2)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("MSE:", mean_squared_error(y, predictions))