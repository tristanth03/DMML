import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tools import load_iris, split_train_test

# Section 1.1: Sigmoid Function and its Derivative
def sigmoid(x):
    # To avoid overflow, return 0.0 when x < -100
    return torch.where(x < -100, torch.tensor(0.0, dtype=x.dtype), 1 / (1 + torch.exp(-x)))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# Section 1.2: Perceptron Function
def perceptron(x, w):
    a = torch.dot(x, w)
    s = sigmoid(a)
    return a, s

# Section 1.3: Forward Propagation Function
def ffnn(x, M, K, W1, W2):
    # Input layer with bias term
    z0 = torch.cat((torch.tensor([1.0], dtype=x.dtype), x))
    # Hidden layer computations
    a1 = torch.mv(W1.t(), z0)
    s1 = sigmoid(a1)
    z1 = torch.cat((torch.tensor([1.0], dtype=s1.dtype), s1))
    # Output layer computations
    a2 = torch.mv(W2.t(), z1)
    y = sigmoid(a2)
    return y, z0, z1, a1, a2

# Section 1.4: Backward Propagation Function
def backprop(x, target_y, M, K, W1, W2):
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    # Output layer delta
    delta_k = y - target_y
    # Hidden layer delta
    delta_j = d_sigmoid(a1) * torch.mv(W2[1:, :], delta_k)
    # Gradient matrices
    dE1 = torch.outer(z0, delta_j)
    dE2 = torch.outer(z1, delta_k)
    return y, dE1, dE2

# Section 2.1: Training Function
def train_nn(X_train, t_train, M, K, W1, W2, iterations, eta):
    N = X_train.shape[0]
    E_total = torch.zeros(iterations)
    misclassification_rate = torch.zeros(iterations)
    W1tr = W1.clone()
    W2tr = W2.clone()

    for iter in range(iterations):
        dE1_total = torch.zeros_like(W1tr)
        dE2_total = torch.zeros_like(W2tr)
        total_error = 0.0
        incorrect = 0
        guesses = torch.zeros(N, dtype=torch.long)

        for n in range(N):
            x = X_train[n, :]
            target_y = torch.zeros(K)
            target_y[t_train[n]] = 1.0
            y, dE1, dE2 = backprop(x, target_y, M, K, W1tr, W2tr)
            dE1_total += dE1
            dE2_total += dE2

            # Cross-entropy error
            epsilon = 1e-12  # To prevent log(0)
            y_clamped = torch.clamp(y, epsilon, 1 - epsilon)
            E_n = -torch.sum(target_y * torch.log(y_clamped) + (1 - target_y) * torch.log(1 - y_clamped))
            total_error += E_n

            # Prediction and misclassification
            predicted_class = torch.argmax(y)
            guesses[n] = predicted_class
            if predicted_class != t_train[n]:
                incorrect += 1

        # Update weights
        W1tr -= eta * dE1_total / N
        W2tr -= eta * dE2_total / N

        # Record error and misclassification rate
        E_total[iter] = total_error / N
        misclassification_rate[iter] = incorrect / N

    return W1tr, W2tr, E_total, misclassification_rate, guesses



# initialize the random generator to get repeatable results
torch.manual_seed(4321)
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)



# initialize the random seed to get predictable results
torch.manual_seed(1234)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * torch.rand(D + 1, M) - 1
W2 = 2 * torch.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)

print(W1tr)
print(W2tr)
print(Etotal)
print(misclassification_rate)
