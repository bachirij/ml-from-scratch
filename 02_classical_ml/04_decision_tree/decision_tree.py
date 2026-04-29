import numpy as np

# Node class to represent each node in the tree
class Node:
    def __init__(self, feature, threshold, left, right, value=None, proba=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Class label for leaf nodes (None for internal nodes)
        self.proba = proba # Class probabilities for leaf nodes (None for internal nodes)

# DecisionTreeClassifier class
class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, max_features=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.
        """
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """
        Predict the class labels for a set of examples.
        """
        return np.array([self._predict_one(self.tree, x) for x in X])
    
    def _entropy(self, y):
        """
        Calculate the entropy of a set of labels.
        """
        if len(y) == 0:
            return 0
        p = np.mean(y)
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def _gini(self, y):
        """
        Calculate the Gini impurity of a set of labels.
        """
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 1 - p**2 - (1 - p)**2
    
    def _impurity(self, y):
        """
        Calculate the impurity of a set of labels based on the chosen criterion.
        """
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'.")
    
    def _information_gain(self, y, left_indices, right_indices):
        """
        Calculate the information gain from a potential split.
        """
        parent_impurity = self._impurity(y)
        left_impurity = self._impurity(y[left_indices])
        right_impurity = self._impurity(y[right_indices])
        
        weight_left = len(left_indices) / len(y)
        weight_right = len(right_indices) / len(y)
        
        gain = parent_impurity - (weight_left * left_impurity + weight_right * right_impurity)
        return gain
    
    def _best_threshold(self, X, y, feature):
        """
        Find the best threshold for a given feature that maximizes information gain.
        """
        thresholds = np.unique(X[:, feature])
        best_gain = -1
        best_threshold = None
        
        for threshold in thresholds:
            left_indices = np.where(X[:, feature] <= threshold)[0]
            right_indices = np.where(X[:, feature] > threshold)[0]
            
            if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                continue
            
            gain = self._information_gain(y, left_indices, right_indices)
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.
        """
        best_feature = None
        best_threshold = None
        best_gain = -1
        
        n_features = X.shape[1]
        features = np.arange(n_features)
        
        if self.max_features is not None:
            features = np.random.choice(features, self.max_features, replace=False)
        
        for feature in features:
            threshold, gain = self._best_threshold(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = np.round(np.mean(y))  # Majority class for binary classification
            return Node(feature=None, threshold=None, left=None, right=None, value=leaf_value, proba=np.mean(y))
        
        # Find the best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            leaf_value = np.round(np.mean(y))  # Majority class for binary classification
            return Node(feature=None, threshold=None, left=None, right=None, value=leaf_value, proba=np.mean(y))
        
        # Split the data
        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]
        
        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def _predict_one(self, node, x):
        """
        Predict the class label for a single example by traversing the tree.
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)
        
    def _predict_proba_one(self, node, x):
        """
        Predict the class probabilities for a single example by traversing the tree.
        """
        if node.proba is not None:
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(node.left, x)
        else:
            return self._predict_proba_one(node.right, x)

    def predict_proba(self, X):
        """
        Predict the class probabilities for a set of examples.
        """
        return np.array([self._predict_proba_one(self.tree, x) for x in X])
        
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the decision tree
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, predictions))