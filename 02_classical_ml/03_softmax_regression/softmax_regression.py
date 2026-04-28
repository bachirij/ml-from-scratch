import numpy as np

# Softmax Regression class
class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

        self.losses = [] # to store loss values during training
        self.weight_history = [] # to store weights values during training
        self.bias_history = [] # to store bias values during training

    def fit(self, X, y):
        """
        Train the softmax regression model using gradient descent.
        """
        # One hot encoding of y
        num_classes = len(np.unique(y)) # number of unique classes, here it is 3 for the iris dataset
        y_one_hot = np.zeros((y.shape[0], num_classes))
        for i in range(y.shape[0]):
            y_one_hot[i, y[i]] = 1

        # Initialize parameters
        num_features = X.shape[1] 
        self.weights = np.random.rand(num_features, num_classes) * 0.01 
        self.bias = np.zeros((1, num_classes)) 

        # Gradient descent
        for _ in range(self.num_iterations):
            y_hat = self._softmax(np.dot(X, self.weights) + self.bias)
            self._update_parameters(X, y_one_hot, y_hat)

            # Update loss and parameter history for monitoring
            loss = self._compute_loss(y_one_hot, y_hat)
            self.losses.append(loss)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

    def predict(self, X):
        """
        Predict categorical labels for the input data.
        """
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        """
        Compute the predicted probabilities for each categorical class.
        """
        return self._softmax(X @ self.weights + self.bias)

    def _softmax(self, z):
        """
        Compute the softmax function.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # subtract max for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _update_parameters(self, X, y, y_hat):
        """
        Update weights and bias using gradient descent.
        """
        m = y.shape[0]
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y, axis=0, keepdims=True)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def _compute_loss(self, y, y_hat):
        """
        Compute the categorical cross-entropy loss.
        """
        m = y.shape[0] # number of samples
        # Clip y_pred to prevent log(0) which is undefined
        y_pred_clipped = np.clip(y_hat, 1e-15, None)
        loss = -np.sum(y * np.log(y_pred_clipped)) / m
        return loss
    
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = SoftmaxRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Final loss: {model.losses[-1]:.4f}")