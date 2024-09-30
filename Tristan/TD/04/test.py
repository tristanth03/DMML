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
    # y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    # print(ffnn(x, M, K, W1, W2))
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

    # y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    # print(y)
    # print(dE1)
    # print(dE2)
    # initialize the random seed to get predictable results
# initialize the random seed to get predictable results
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
   

    guesses = test_nn(test_features[:20,:],M,K,W1tr,W2tr)
    print(guesses)
    print(test_targets[:20])
    plot_loss(Etotal)
    plot_misclassification_rate(misclassification_rate)
    print(accuracy(misclassification_rate))

#### 
    print("Guess")
    print(guesses)
    print("Target")
    print(test_targets[:20])
    c_mat = confusion_matrix(K,test_targets[:20],guesses)
    print("Confusion Matrix")
    print(c_mat)