import numpy as np

def sigmoid(x):    
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, shape=[10, 20, 4]):
        self.weights = []
        self.bias = []
        
        # Initialize weights and biases
        # [100, 10] x [10, 20], -> [100, 20]  
        # [M, N] x [N, P] = [M, P]
        
        for i in range(len(shape) - 1):
            self.weights.append(np.random.rand(shape[i], shape[i+1]))
            self.bias.append(np.random.rand(shape[i+1]))
    
    def loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)
    
    '''
    Inputs: np.array of shape (Inputs, N)
    Returns: np.array of shape (P, Outputs)
    '''
    def forward(self, inputs):
        X = inputs
        
        for i in range(len(self.weights)-1):
            X = np.dot(X, self.weights[i]) + self.bias[i]
            X = sigmoid(X)
            
        return sigmoid(np.dot(X, self.weights[-1]) + self.bias[-1])