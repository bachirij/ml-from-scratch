# linear_regression.py
import numpy as np

# Linear Regression class
class LinearRegression:
    def __init__(
            self, 
            learning_rate=0.01, 
            n_iters=1000,
            batch_size=None,
            regularization=None,   # None | "l1" (Lasso) | "l2" (Ridge)
            lambda_reg=0.0): 
        
        self.lr = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.w = None # weights
        self.b = None # bias

        self.losses = [] # to store loss values during training
        self.weight_history = [] # to store weights values during training
        self.bias_history = [] # to store bias values during training

    # Fit method to train the model using the training data
    def fit(self, X, y):
        # Ensure correct shape for y
        y = y.ravel()  
        
        # Initialise dimensions
        n_samples, n_features = X.shape

        # Initialise weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # If batch_size is None, then full batch gradient descent
        if self.batch_size is None:
            self.batch_size = n_samples

        # Training loop
        for _ in range(self.n_iters):

            # Shuffle dataset at each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self._update_parameters(X_batch, y_batch)

            # Compute full loss for monitoring
            y_pred = self.predict(X)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

    # Predict method to make predictions using the learned weights and bias
    def predict(self, X):
        return np.dot(X, self.w) + self.b # y = Xw + b
    
    # Private method to update weights and bias using gradient descent
    def _update_parameters(self, X, y):
        m = X.shape[0]

        y_pred = self.predict(X)
        error = y_pred - y

        # Base gradients
        dw = (2 / m) * np.dot(X.T, error)
        db = (2 / m) * np.sum(error)

        # Add regularization
        if self.regularization == "l2":
            dw += 2 * self.lambda_reg * self.w

        elif self.regularization == "l1":
            dw += self.lambda_reg * np.sign(self.w)

        # Update
        self.w -= self.lr * dw
        self.b -= self.lr * db

    # Private method to compute the loss (MSE + regularization)
    def _compute_loss(self, y, y_pred):
        mse = np.mean((y - y_pred) ** 2)

        if self.regularization == "l2":
            reg_term = self.lambda_reg * np.sum(self.w ** 2)

        elif self.regularization == "l1":
            reg_term = self.lambda_reg * np.sum(np.abs(self.w))

        else:
            reg_term = 0

        return mse + reg_term

if __name__ == "__main__":
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = LinearRegression(learning_rate=0.1, n_iters=10)
    model.fit(X, y)
    print("Weights:", model.w)
    print("Bias:", model.b)