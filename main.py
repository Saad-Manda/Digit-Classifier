import numpy as np
import struct
# import tensorflow as tf

# def load_mnist_images(filename):
#     with gzip.open(filename, 'rb') as f:
#         _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
#         data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
#         return data.T / 255.0  # Normalize pixel values to [0, 1]

# def load_mnist_labels(filename):
#     with gzip.open(filename, 'rb') as f:
#         _, num_items = struct.unpack(">II", f.read(8))
#         labels = np.frombuffer(f.read(), dtype=np.uint8)
#         return np.eye(10)[labels].T  # Convert labels to one-hot encoding

def load_mnist_images(filename):
    with open(filename, 'rb') as f:  # Use open() instead of gzip.open()
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return data.T / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:  # Use open() instead of gzip.open()
        _, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return np.eye(10)[labels].T

    
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(0)
    # W1 = np.random.randn(hidden_size, input_size)*0.01  #Weight for hidden layer
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2./input_size)
    b1 = np.random.randn(hidden_size, 1)
    # W2 = np.random.randn(output_size, hidden_size)*0.01  #Weight for output layer
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2./hidden_size)
    b2 = np.random.randn(output_size, 1) #Weight for output layer
    return {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

# def sigmoid(z):
#     return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(Z):
    return Z > 0

def softmax(z):
    exp_z = np.exp(z-np.max(z))
    ans = exp_z / exp_z.sum(axis = 0, keepdims = True)
    return ans

def forward_propagation(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = np.dot(W1, X) + b1
    # A1 = sigmoid(Z1)
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_loss(Y, A2):
    m = Y.shape[1]
    loss = -np.sum(Y*np.log(A2 + 1e-8))/m
    return loss

def back_propagation(X, Y, params, cache):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1, A1, Z2, A2 = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"]

    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2,  axis=1, keepdims=True) / m

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradient = {"dW1":dW1, "db1": db1, "dW2":dW2, "db2":db2}
    return gradient

def update_parameters(params, gradient, learning_rate):
    params["W1"] -= learning_rate*gradient["dW1"]
    params["b1"] -= learning_rate*gradient["db1"]
    params["W2"] -= learning_rate*gradient["dW2"]
    params["b2"] -= learning_rate*gradient["db2"]
    return params
    
def train_neural_network(X, Y, input_size, hidden_size, output_size, learning_rate, epochs, batch_size = 32):
    params = initialize_parameters(input_size, hidden_size, output_size)

    for i in range(epochs):
        for j in range(0, X.shape[1], batch_size):
            X_batch = X[:, j:j+batch_size]
            Y_batch = Y[:, j:j+batch_size]
            
            A2, cache = forward_propagation(X_batch, params)
            gradients = back_propagation(X_batch, Y_batch, params, cache)
            params = update_parameters(params, gradients, learning_rate)

        if i % 10 == 0:
            A2, _ = forward_propagation(X, params)
            loss = compute_loss(Y, A2)
            print(f"Epoch = {i}, loss = {loss}")

    return params

def predict(X, params):
    A2, _ = forward_propagation(X, params)
    predictions = np.argmax(A2, axis=0)
    return predictions
    


# Load training and test data from the MNIST dataset
X_train = load_mnist_images('train-images.idx3-ubyte')
Y_train = load_mnist_labels('train-labels.idx1-ubyte')
X_test = load_mnist_images('t10k-images.idx3-ubyte')
Y_test = load_mnist_labels('t10k-labels.idx1-ubyte')


# Define network parameters
input_size = 784  # 28x28 pixels
hidden_size = 128  # Hidden layer neurons
output_size = 10  # Digits from 0 to 9
learning_rate = 0.01
epochs = 50

# Train the model with MNIST data
trained_params = train_neural_network(X_train, Y_train, input_size, hidden_size, output_size, learning_rate, epochs)

# Test the trained model
predictions = predict(X_test, trained_params)
accuracy = np.mean(np.argmax(Y_test, axis=0) == predictions) * 100
print(f"Test Accuracy: {accuracy:.2f}%")