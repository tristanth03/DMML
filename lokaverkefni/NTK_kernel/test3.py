import torch 
import torch.nn as nn
import torch.linalg
import torch.nn.functional as F
import torch.optim as optim



if __name__ == "__main__":
    torch.manual_seed(1234)
    x1 = [1,-1]
    x2 = [2,0]
    x3 = [3,-3]
    x4 = [4,-4]

    X = torch.tensor([x1,x2,x3,x4],dtype=torch.float64)
    T = torch.exp(X)

    N,D = X.shape
    K = T.shape[1]
    
  
    M1 = 10_000
    model = nn.Sequential(
        nn.Linear(D, M1, bias=True),
        nn.ReLU(),
        nn.Linear(M1, K, bias=True)
        ).double() # double for precision measure

        # Number of epochs
    num_epochs = 10000

    # Define loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Define the optimizer (Stochastic Gradient Descent, equivalent to normal GD)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)  # Learning rate for normal GD

    # Training loop with Full Batch Gradient Descent
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute model output for the entire dataset
        predictions = model(X)

        # Compute loss between predictions and true values (on the entire dataset)
        loss = criterion(predictions, T)

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Print progress occasionally
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.16f}")

    # Evaluation after training (optional)
    model.eval()
    with torch.no_grad():
        final_predictions = model(X)
        final_loss = criterion(final_predictions, T)
        print(f"Final Loss after Training: {final_loss.item():.16f}")