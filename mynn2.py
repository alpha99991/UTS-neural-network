import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(Z, 0)


def deriv_relu(Z):
    Z[Z > 0] = 1
    Z[Z < 0] = 0
    return Z


def layer_sizes(X, Y):

    n_x = X.shape[0]  # size of input layer
    n_y = Y.shape[0]  # size of output layer

    return (n_x, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):

    m = Y.shape[1]  # number of example

    cost = np.sum(np.multiply((A2 - Y), (A2 - Y))) * (1 / m)

    return cost

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve A1 and A2 from dictionary "cache".

    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dA2 = A2 - Y
    dZ2 = np.multiply(np.multiply(A2, 1 - A2), dA2)
    dW2 = 1 / m * (np.dot(dZ2, A1.T))
    db2 = 1 / m * (np.sum(dZ2, axis=1, keepdims=True))
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(np.multiply(A1, 1 - A1), dA1)
    dW1 = 1 / m * (np.dot(dZ1, X.T))
    db1 = 1 / m * (np.sum(dZ1, axis=1, keepdims=True))

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate=0.5):

    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters

def nn_model(X, Y, X_test, Y_test, n_h, num_iterations=1000):
    n_x, n_y = layer_sizes(X, Y)

    # Initialize parameters

    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        # Compute test cost for this iteration
        A2, cache = forward_propagation(X_test, parameters)
        test_cost = compute_cost(A2, Y_test, parameters)

        iter_plot.append(i)
        cost_plot.append(cost)
        test_cost_plot.append(test_cost)

    return parameters


cost_plot = []
test_cost_plot = []
iter_plot = []

if __name__ == "__main__":
    np.random.seed(67465)

    minmaxscalar = MinMaxScaler(feature_range=(0.1, 0.9))

    data = pd.read_excel("D:/PROJECTS/weld_data.xlsx")

    X_train = data.iloc[0:46, 1:9]
    X_test = data.iloc[46:53, 1:9]

    Y_train = data.iloc[0:46, 9:10]
    Y_test = data.iloc[46:53, 9:10]

    X_train = minmaxscalar.fit_transform(X_train)

    X_test = minmaxscalar.fit_transform(X_test)

    Y_train = minmaxscalar.fit_transform(Y_train)

    Y_test = minmaxscalar.fit_transform(Y_test)

    # print(X_train)

    X_train = X_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    Y_train = Y_train.T

    parameters = nn_model(X_train, Y_train, X_test, Y_test, 8, num_iterations=10000)

    print(cost_plot[-1])

    plt.plot(iter_plot, cost_plot)
    plt.plot(iter_plot, test_cost_plot)
    plt.xlabel("ITER")
    plt.ylabel("LOSS")
    plt.show()

