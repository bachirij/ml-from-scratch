import numpy as np

# Logistic Regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

        self.losses = [] # to store loss values during training
        self.weight_history = [] # to store weights values during training
        self.bias_history = [] # to store bias values during training

    def fit(self, X, y):
        # Initialize parameters
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient descent
        for _ in range(self.max_iterations):
            y_hat = self._sigmoid(np.dot(X, self.weights) + self.bias)
            self._update_parameters(X, y, y_hat)

            # Update loss and parameter history for monitoring
            loss = self._compute_loss(y, y_hat)
            self.losses.append(loss)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

    def predict(self, X):
        y_hat = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return (y_hat >= 0.5).astype(int)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_hat):
        m = y.shape[0]
        loss = (-1/m) * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        return loss

    def _update_parameters(self, X, y, y_hat):
        m = y.shape[0]
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

if __name__ == "__main__":
    # Example usage
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1]) # OR logic gate
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)