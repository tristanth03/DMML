# Author: Tristan Thordarson
# Date: 26.09.2024
# Project: 04
# Acknowledgements: 
    # [1] https://pytorch.org/docs/stable/generated/torch.where.html
    # [2] https://www.tutorialspoint.com/how-to-join-tensors-in-pytorch#:~:text=We%20can%20join%20two%20or,used%20to%20stack%20the%20tensors
    # [3] https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#

from typing import Union
import torch

from tools import load_iris, split_train_test


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the sigmoid of x
    '''
    a = torch.where(x>=-100,x,0.0) # torch method to check element wise on the tensor
    sigmoid = 1/(1+torch.exp(-x))
    return sigmoid


def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    s = sigmoid(x)
    d_sigmoid = s*(1-s)
    return d_sigmoid

def perceptron(
    x: torch.Tensor,
    w: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    z_pre = x # z^(l-1) = z^(0) = x
    a = torch.dot(w,z_pre)
    z = sigmoid(a)
    return a,z


def ffnn(
    x: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    b = torch.Tensor([1.0]) # x0, bias
    a0 = x # input layer (input*indentity) [1x(D)]
    D = x.shape[0]
    z0 = torch.cat((b,a0)) # [1x(D+1)]  # there is not activation function on the input layer & cat: concatenate (join)
    W1 = W1 # weights from input to hidden layer [(D+1)xM]
    a1 = torch.matmul(z0,W1) # [1xM]
    z1 = torch.cat((b,sigmoid(a1)))
    W2 = W2 # weights from hidden layer to output layer [(M+1)xK]
    a2 = torch.matmul(z1,W2) 
    y = sigmoid(a2)

    return y,z0,z1,a1,a2
    
def backprop(
    x: torch.Tensor,
    target_y: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''

    # 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    # 2
    delta_K = y-target_y
    # 3
    d_sigm = d_sigmoid(a1)
    delta_J = d_sigm*(torch.matmul(W2[1:,:],delta_K)) # removing the bias, assuming its the first row
    # !!!!! Skrifa smá línu samantekt hér!!!
    # 4
    # pointless 
    # 5
    dE1 = torch.matmul(z0.unsqueeze(1),delta_J.unsqueeze(0))
    dE2 = torch.matmul(z1.unsqueeze(1),delta_K.unsqueeze(0))

    return y, dE1, dE2

def train_nn(
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor,
    iterations: int,
    eta: float
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    epochs = iterations
    Loss = []
    D = X_train.shape[1]
    for epoch in range(epochs):
        # This training algorithm is memoryless in a sense that we do not hold on to older weights
        if epoch == 0:
            W1 = W1
            W2 = W2
            dE1 = torch.zeros(D+1,M)
            dE2 = torch.zeros(M+1,K)
        # implementa að stoppa ef loss hækkar meira en það var í upphafi

        W1 = W1-eta*dE1
        W2 = W2-eta*dE2

        y, dE1, dE2 = backprop(X_train[1],t_train[1],M,K,W1,W2)
        
        print(y)
        



def test_nn(
    X: torch.Tensor,
    M: int,
    K: int,
    W1: torch.Tensor,
    W2: torch.Tensor
) -> torch.Tensor:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    ...


def extra():
    # l = w.shape[0]

    # 1
    # ffnn with k_i hidden nodes per hidden layer 
    # with l-many hidden layers

    # 2
    # Manual cross-entropy
    # torch.log(vector)
    # Loss = torch.nn.CrossEntropyLoss()  

    pass



if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    # initialize the random generator to get repeatable results
    torch.manual_seed(1234)

    # Take one point:
    x = train_features[0, :]
    K = 3  # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    print(ffnn(x, M, K, W1, W2))
    # initialize the random generator to get repeatable results
    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    # initialize random generator to get predictable results
    torch.manual_seed(42)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    x = features[0, :]

    # create one-hot target for the feature
    target_y = torch.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    print(y)
    print(dE1)
    print(dE2)
    # initialize the random seed to get predictable results
    torch.manual_seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * torch.rand(D + 1, M) - 1
    W2 = 2 * torch.rand(M + 1, K) - 1
    # W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    #     train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)

    # y, dE1, dE2 = backprop(train_features[:20, :], train_targets[:20],M,K,W1,W2)

  
    
    
#### 