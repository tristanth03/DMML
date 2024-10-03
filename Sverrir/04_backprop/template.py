from typing import Union
import torch

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
    ...


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
    ...


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