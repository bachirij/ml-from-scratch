import numpy as np

# Node class to represent each node in the tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature # Feature index to split on
        self.threshold = threshold # Threshold value for the split
        self.left = left # Left child node
        self.right = right # Right child node
        self.value = value # Predicted value for leaf nodes (None for internal nodes)
        self.is_leaf = value is not None # Flag to indicate if the node is a leaf

# RegressionTree class
class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=20, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the regression tree to the training data.
        """
        self.root = self._build_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target values for a set of examples.
        """
        return np.array([self._predict_one(self.root, x) for x in X])
    
    def _compute_mse(self, y):
        """
        Compute the Mean Squared Error of a set of target values.
        """
        n_examples = y.size
        return (1/n_examples) * np.sum((y - np.mean(y)) ** 2)
    
    def _compute_midpoints(self, X_feature):
        """
        Compute threshold candidatures (midpoints between sorted unique values) for each feature.
        """
        thresholds = np.unique(X_feature)[:-1] + np.diff(np.unique(X_feature)) / 2
        return thresholds
    
    def _compute_split_mse(self, X_feature, y, threshold):
        """
        Compute the MSE of a potential split.
        """
        left_indices = X_feature <= threshold
        right_indices = X_feature > threshold
        
        # If the split is empty, return +infinity (because we want to minimize the weighted MSE)
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.inf
        
        mse_left = self._compute_mse(y[left_indices])
        mse_right = self._compute_mse(y[right_indices])
        
        weighted_mse = (np.sum(left_indices) * mse_left + np.sum(right_indices) * mse_right) / len(y)
        
        return weighted_mse
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on based on the lowest MSE.
        """
        best_gain = np.inf
        best_feature = None
        best_thresh = None

        n_features = X.shape[1]
        features = np.arange(n_features)

        # If max_features is specified, randomly select a subset of features
        if self.max_features is not None:
            features = np.random.choice(features, size=self.max_features, replace=False)

        # For each feature, find the best threshold and gain
        for feature in features:
            X_feature = X[:, feature]
            thresholds = self._compute_midpoints(X_feature)

            for threshold in thresholds:
                gain = self._compute_split_mse(X_feature, y, threshold)

                if gain < best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = threshold

        return (best_feature, best_thresh)
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the regression tree.
        """
        
        # Stopping criteria
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=np.mean(y)) # return the mean of the target values as the prediction for this leaf node

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=np.mean(y))

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        # If the split is empty, return the mean of the target values
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return Node(value=np.mean(y))

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def _predict_one(self, tree, x):
        """
        Predict the target value for a single example by traversing the tree.
        """
        # Checks if the tree is a leaf node (True) and returns the prediction
        if tree.is_leaf:
            return tree.value
        
        # Unpack the tree node
        feature = tree.feature
        threshold = tree.threshold
        left_subtree = tree.left
        right_subtree = tree.right

        # For each sample, traverse the tree according to the feature and threshold values until reaching a leaf node
        if x[feature] <= threshold:
            return self._predict_one(left_subtree, x)
        else:
            return self._predict_one(right_subtree, x)
    

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, root_mean_squared_error

    # Load the diabetes dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the regression tree on the training data
    tree = RegressionTree(max_depth=5, min_samples_split=10)
    tree.fit(X_train, y_train)

    # Predict on the test set
    y_pred = tree.predict(X_test)

    # Evaluate the model
    print("Test R2 score:", r2_score(y_test, y_pred))
    print("Test RMSE:", root_mean_squared_error(y_test, y_pred))

    from sklearn.tree import DecisionTreeRegressor

    sklearn_tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
    sklearn_tree.fit(X_train, y_train)
    y_pred_sklearn = sklearn_tree.predict(X_test)

    print("Sklearn Test R2 score:", r2_score(y_test, y_pred_sklearn))
    print("Sklearn Test RMSE:", root_mean_squared_error(y_test, y_pred_sklearn))