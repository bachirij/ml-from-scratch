# linear_regression.py
import numpy as np

# Linear Regression class
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000): 
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None # weights
        self.b = None # bias
        self.losses = [] # to store loss values during training
        self.weight_history = [] # to store weights values during training
        self.bias_history = [] # to store bias values during training

    # Fit method to train the model using the training data
    def fit(self, X, y):
        y = y.ravel()  # Ensure correct shape
        
        # Initialise dimensions
        n_samples, n_features = X.shape

        # Initialise weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for _ in range(self.n_iters):
            # Forward pass: compute predictions
            y_pred = self.predict(X)

            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
        
            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y)) # gradient of weights
            db = (2 / n_samples) * np.sum(y_pred - y) # gradient of bias

            # Update weights and bias
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Store weights and bias history for visualization
            self.weight_history.append(self.w.copy())
            self.bias_history.append(self.b)

    # Predict method to make predictions using the learned weights and bias
    def predict(self, X):
        return np.dot(X, self.w) + self.b # y = Xw + b
    
    # Internal method to compute the mean squared error loss
    def _compute_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

if __name__ == "__main__":
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = LinearRegression(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    print("Weights:", model.w)
    print("Bias:", model.b)