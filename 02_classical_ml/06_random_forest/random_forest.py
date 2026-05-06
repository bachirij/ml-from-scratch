import numpy as np
from regression_tree import RegressionTree
from decision_tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, r2_score

# RandomForestClassifier class
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_features=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.forest = None

    def fit(self, X, y):
        """
        Fit the random forest to the training data.
        """
        # Store the number of features (of feature_importance())
        self.n_features = X.shape[1]

        # List to store trees
        forest = []

        # Loop over each tree
        for _ in range(self.n_estimators):
            X_sample, y_sample, oob_indices = self._boostrap_sampling(X, y)
            model = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            model.fit(X_sample, y_sample)
            forest.append((model, oob_indices))

        self.forest = forest
        return self
    
    def predict(self, X):
        """
        Predict the class labels for a set of examples.
        """
        # List to store predictions from each tree
        tree_predictions = []

        # Loop over each tree in the forest
        for model, _ in self.forest:
            tree_predictions.append(model.predict(X))

        # Stack predictions from all trees into a 2D array (matrix)
        tree_predictions = np.stack(tree_predictions, axis=1) # shape (n_samples, n_estimators)
        tree_predictions = tree_predictions.astype(int)

        # Majority vote: for each sample, find the most common prediction among the trees
        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_predictions)

        return predictions
    
    def oob_score(self, X, y):
        """
        Calculate the Out-of-Bag (OOB) accuracy of the random forest.
        """
        # Initialize an empty list to store OOB predictions for each sample
        n_samples = X.shape[0]
        oob_predictions = [[] for _ in range(n_samples)]

        # For each tree in the forest, get the OOB indices and make predictions for those samples
        for model, oob_indices in self.forest:
            predictions = model.predict(X[oob_indices]).astype(int)
            for idx, pred in zip(oob_indices, predictions):
                oob_predictions[idx].append(pred)

        # For each sample, find the most common prediction among the trees that did not use it for training
        final_oob_predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Get the predictions from trees that did not use this sample for training
            tree_preds = np.array(oob_predictions[i])
            if len(tree_preds) > 0: # if there are any OOB predictions for this sample
                final_oob_predictions[i] = np.bincount(tree_preds).argmax() # majority vote

        # Calculate OOB accuracy
        oob_accuracy = accuracy_score(y, final_oob_predictions)
        return oob_accuracy
    
    def feature_importance(self):
        """
        Calculate feature importance based on the frequency of feature usage in the trees.
        """
        # Initialize an array to store feature importance scores
        importance_scores = np.zeros(self.n_features)

        # Recursive function to traverse the tree and update importance scores
        def _traverse(node):
            if node.value is not None:  # leaf, we stop traversing
                return
            importance_scores[node.feature] += 1
            _traverse(node.left)
            _traverse(node.right)

        # For each tree in the forest, check if the node is a leaf node or not, if not, get the feature index used for splitting and update the importance score for that feature
        for model, _ in self.forest:
            # Check if the node is a leaf node
            _traverse(model.tree)

        # Normalize importance scores
        importance_scores /= np.sum(importance_scores)

        return importance_scores
    
    def _boostrap_sampling(self, X, y):
        """
        Perform bootstrap sampling to create a random subset of the training data for each tree.
        Returns the sampled data and the indices of the out-of-bag (OOB) samples.
        """
        # Number of training exemples
        n_examples = X.shape[0]

        # Sample indices with replacement
        sampled_indices = np.random.choice(n_examples, n_examples, replace=True)

        # Get the sampled data
        X_sample = X[sampled_indices]
        y_sample = y[sampled_indices]

        # Get the out-of-bag (OOB) indices
        oob_indices = np.setdiff1d(np.arange(n_examples), sampled_indices) # return the indices that are not in the sampled_indices

        return X_sample, y_sample, oob_indices


# RandomForestRegressor class
class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_features=None, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.forest = None

    def fit(self, X, y):
        """
        Fit the random forest to the training data.
        """
        # Store the number of features (of feature_importance())
        self.n_features = X.shape[1]

        # List to store trees
        forest = []

        # Loop over each tree
        for _ in range(self.n_estimators):
            X_sample, y_sample, oob_indices = self._boostrap_sampling(X, y)
            model = RegressionTree(max_depth=self.max_depth, max_features=self.max_features, min_samples_split=self.min_samples_split)
            model.fit(X_sample, y_sample)
            forest.append((model, oob_indices))

        self.forest = forest
        return self
    
    def predict(self, X):
        """
        Predict the class labels for a set of examples.
        """
        # List to store predictions from each tree
        tree_predictions = []

        # Loop over each tree in the forest
        for model, _ in self.forest:
            tree_predictions.append(model.predict(X))

        # Stack predictions from all trees into a 2D array (matrix)
        tree_predictions = np.stack(tree_predictions, axis=1) # shape (n_samples, n_estimators)

        # Mean vote: for each sample, find the mean prediction among the trees
        predictions = np.apply_along_axis(lambda x: np.mean(x), axis=1, arr=tree_predictions)

        return predictions
    
    def oob_score(self, X, y):
        """
        Calculate the Out-of-Bag (OOB) r2_score of the random forest.
        """
        # Initialize an empty list to store OOB predictions for each sample
        n_samples = X.shape[0]
        oob_predictions = [[] for _ in range(n_samples)]

        # For each tree in the forest, get the OOB indices and make predictions for those samples
        for model, oob_indices in self.forest:
            predictions = model.predict(X[oob_indices])
            for idx, pred in zip(oob_indices, predictions):
                oob_predictions[idx].append(pred)

        # For each sample, find the most common prediction among the trees that did not use it for training
        final_oob_predictions = np.zeros(n_samples)
        for i in range(n_samples):
            # Get the predictions from trees that did not use this sample for training
            tree_preds = np.array(oob_predictions[i])
            if len(tree_preds) > 0: # if there are any OOB predictions for this sample
                final_oob_predictions[i] = np.mean(tree_preds) # mean vote

        # Calculate OOB r2_score
        oob_r2_score = r2_score(y, final_oob_predictions)
        return oob_r2_score

    def feature_importance(self):
        """
        Calculate feature importance based on the frequency of feature usage in the trees.
        """
        # Initialize an array to store feature importance scores
        importance_scores = np.zeros(self.n_features)

        # Recursive function to traverse the tree and update importance scores
        def _traverse(node):
            if node.is_leaf:  # leaf, we stop traversing
                return
            importance_scores[node.feature] += 1
            _traverse(node.left)
            _traverse(node.right)

        # For each tree in the forest, check if the node is a leaf node or not, if not, get the feature index used for splitting and update the importance score for that feature
        for model, _ in self.forest:
            # Check if the node is a leaf node
            _traverse(model.root)

        # Normalize importance scores
        importance_scores /= np.sum(importance_scores)

        return importance_scores
    
    def _boostrap_sampling(self, X, y):
        """
        Perform bootstrap sampling to create a random subset of the training data for each tree.
        Returns the sampled data and the indices of the out-of-bag (OOB) samples.
        """
        # Number of training exemples
        n_examples = X.shape[0]

        # Sample indices with replacement
        sampled_indices = np.random.choice(n_examples, n_examples, replace=True)

        # Get the sampled data
        X_sample = X[sampled_indices]
        y_sample = y[sampled_indices]

        # Get the out-of-bag (OOB) indices
        oob_indices = np.setdiff1d(np.arange(n_examples), sampled_indices) # return the indices that are not in the sampled_indices

        return X_sample, y_sample, oob_indices

if __name__ == "__main__":
    # Example usage


    # Classification
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the random forest
    model = RandomForestClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, predictions))

    # Compute OBB accuracy
    oob_accuracy = model.oob_score(X_train, y_train)
    print("OOB Accuracy:", oob_accuracy)

    # Print feature importance    
    features = [f"Feature {i}" for i in range(X.shape[1])]
    importances = model.feature_importance()
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"{features[i]}: {importances[i]:.4f}")


    # Regression
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import r2_score, root_mean_squared_error

    # Load the diabetes dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the random forest on the training data
    tree = RandomForestRegressor(max_depth=5, min_samples_split=10)
    tree.fit(X_train, y_train)

    # Predict on the test set
    y_pred = tree.predict(X_test)

    # Evaluate the model
    print("Test R2 score:", r2_score(y_test, y_pred))
    print("Test RMSE:", root_mean_squared_error(y_test, y_pred))

    # Compute OBB r2_score
    oob_r2_score = tree.oob_score(X_train, y_train)
    print("OOB r2 score:", oob_r2_score)

    # Print feature importance    
    features = [f"Feature {i}" for i in range(X.shape[1])]
    importances = tree.feature_importance()
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"{features[i]}: {importances[i]:.4f}")
