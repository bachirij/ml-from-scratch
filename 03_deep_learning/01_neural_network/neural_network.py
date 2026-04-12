import numpy as np

def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def _relu(z):
    return np.maximum(0, z)

def _relu_derivative(z):
    return (z > 0).astype(float)
    
#  Activation functions and their derivatives
ACTIVATIONS = {
    'relu':    (_relu,    _relu_derivative),
    'sigmoid': (_sigmoid, lambda z: _sigmoid(z) * (1 - _sigmoid(z))),
}

# Layer class
class Layer:
    def __init__(self, n_output, activation):
        self.n_output = n_output
        self.activation = activation
        self.g, self.g_prime = ACTIVATIONS[activation]

    def initialize(self, n_input):
        # Initialize weights and biases
        W = np.random.randn(self.n_output, n_input) * np.sqrt(2. / n_input) # Xavier initialization
        b = np.zeros((self.n_output, 1))

        # Store weights, biases and input size for backward pass
        self.W = W
        self.b = b
        self.n_input = n_input

    def forward(self, A_prev):
        Z = self.W @ A_prev + self.b
        A = self.g(Z)
        
        # Stores A_prev and Z for the backward pass
        self.A_prev = A_prev
        self.Z = Z

        # Return the activated output
        return A

    def backward(self, dA):
        delta = dA * self.g_prime(self.Z) # delta = dA * activation_derivative(Z)
        dW = delta @ self.A_prev.T # dW = delta @ A_prev.T
        db = np.mean(delta, axis=1, keepdims=True) # db = mean(delta, axis=1, keepdims=True)
        dA_prev = self.W.T @ delta # dA_prev = W.T @ delta

        # Store dW and db for the update step
        self.dW = dW
        self.db = db

        # Return dA_prev for the previous layer
        return dA_prev

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

# Neural Network class
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []

    def add(self, layer):
        # Add a layer to the network and initialize it
        self.layers.append(layer)

    def forward(self, X):
        # boucle sur self.layers
        A = X.T # (features, samples)
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
        # Compute dA from the loss function (BCE)
        dA = -(y / y_hat) + ((1 - y) / (1 - y_hat))
        
        # Reverse loop over each layer to perform backpropagation
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, learning_rate):
        # Loop over each layer to update weights and biases
        for layer in self.layers:
            layer.update(learning_rate)

    def compute_loss(self, y, y_hat):
        # BCE
        m = y.shape[1]
        y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9) 
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
        return loss

    def fit(self, X, y, learning_rate, n_iterations):
        # Initialize all layers in the network
        n_input = X.shape[1]
        for layer in self.layers:
            layer.initialize(n_input)
            n_input = layer.n_output

        # Training loop
        for _ in range(n_iterations):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)
            self.backward(y, y_hat)
            self.update(learning_rate)
        
            # Store the loss
            self.loss_history.append(loss) 

    def predict(self, X):
        # Forward pass over all the network
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int)

