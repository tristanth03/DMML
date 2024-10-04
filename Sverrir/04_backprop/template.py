from typing import Union
import torch
import matplotlib.pyplot as plt
from tools import load_iris, split_train_test


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the sigmoid of x
    '''
    a = torch.where(x < -100,0.0, x)
    sigm = 1/(1+torch.exp(-a))
    return sigm


def d_sigmoid(x: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    sigm = sigmoid(x)
    d_sigmoid = sigm * (1-sigm)
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
    a_0 = torch.dot(x,w)
    return (a_0, sigmoid(a_0))
    


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
    b = torch.Tensor([1.0]) # bias
    z0 = torch.cat((b,x)) # [1x(D+1)]
    a1 = torch.matmul(z0,W1) # [1xM]
    z1 = torch.cat((b,sigmoid(a1))) #[1x(M+1)]
    a2 = torch.matmul(z1,W2) # [1xK]
    y = sigmoid(a2) # [1xK]
    
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
    # Action 1
    y,z0,z1,a1,a2 = ffnn(x, M, K, W1, W2)
    # Action 2
    d_k = y - target_y
    # Action 3
    d_j = d_sigmoid(a1)*torch.matmul(W2[1:,:], d_k)
    # Action 4
    dE1 = torch.empty(W1.shape)
    dE2 = torch.empty(W2.shape)
    # Action 5
    dE1 = torch.matmul(z0.unsqueeze(1),d_j.unsqueeze(0))
    dE2 = torch.matmul(z1.unsqueeze(1),d_k.unsqueeze(0)) 

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
    # Action 1 - Initializing necessary variables
    e_total = []
    all_misclassification_rates = []
    D = X_train.shape[1] # dimention of the input data
    N = X_train.shape[0] # nr of data points
    t = torch.zeros(N,K) # t for target value
    t[torch.arange(N),t_train] = 1.0 # so the correct output for each training data point is 1.0, while all other are 0.0

    # Action 2 - for loop for each iteration
    for iteration in range(0,iterations,1):
        # Action 3 Initializing the dE_total and the estimate 
        dE1_total = torch.zeros(D+1,M)
        dE2_total = torch.zeros(M+1, K)
        y_hat = [] # the estimate for y

        for i in range(N):
            # Action 4 - get all the individual gradients and add them to the total.
            y,dE1, dE2 = backprop(X_train[i,:],t[i,:],M,K,W1,W2)
            dE1_total = dE1_total + dE1
            dE2_total = dE2_total + dE2
            y_hat.append(y)

        # Action 5 adjust the W1 and W2
        W1 = W1 - eta*dE1_total/N
        W2 = W2 - eta*dE2_total/N

        y_hat = torch.stack(y_hat)
        guesses = torch.argmax(y_hat,dim=1) # use one hot encoding, it updates always for each iteration so we return the last guess after we havae trained the model.

        # Action 6 - calculating the loss
        loss_N = -torch.sum(t*torch.log(y_hat)+(1-t)*torch.log(1-y_hat),dim=1)
        mean_loss = torch.mean(loss_N) # get the mean loss
        e_total.append(mean_loss) 

        misclassification_rate = torch.sum(guesses != t_train) / N
        all_misclassification_rates.append(misclassification_rate)

    e_total = torch.stack(e_total)
    all_misclassification_rates = torch.stack(all_misclassification_rates)

        # Action 7 - return the the stuff
    return W1, W2, e_total, all_misclassification_rates, guesses



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


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # print(sigmoid(torch.Tensor([0.5])))
    # print(d_sigmoid(torch.Tensor([0.2])))

    # print(perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1])))
    # print(perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4])))

    # initialize random generator to get predictable results

    torch.manual_seed(4321)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
    torch.manual_seed(42)

    # K = 3  # number of classes
    # M = 6
    # D = train_features.shape[1]

    # x = features[0, :]

    # # create one-hot target for the feature
    # target_y = torch.zeros(K)
    # target_y[targets[0]] = 1.0

    # # Initialize two random weight matrices
    # W1 = 2 * torch.rand(D + 1, M) - 1
    # W2 = 2 * torch.rand(M + 1, K) - 1

    # y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    # #
    # print('------')
    # print(y)
    # print('-------')
    # print(dE1)
    # print('-------')
    # print(dE2)
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
    print('------------')
    print(misclassification_rate)
    print(last_guesses)