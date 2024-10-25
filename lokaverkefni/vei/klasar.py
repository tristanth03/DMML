import torch
import torch.nn as nn
from tqdm import tqdm


class FeedForwardNN(nn.Module):
    def __init__(self, x, t, M, activation_func=nn.ReLU(), ntk_normalization=True):
        super(FeedForwardNN, self).__init__()
        self.input_dim = x.shape[1]
        self.output_dim = t.shape[1]
        self.hidden_dims = M
        self.activation_func = activation_func
        self.ntk_normalization = ntk_normalization

        layers = []
        layer_input_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layer = nn.Linear(layer_input_dim, hidden_dim)
            if self.ntk_normalization:
                nn.init.normal_(layer.weight, mean=0, std=1.0 / torch.sqrt(torch.tensor(layer_input_dim, dtype=torch.float32)))
            layers.append(layer)
            layers.append(self.activation_func)
            layer_input_dim = hidden_dim
        final_layer = nn.Linear(layer_input_dim, self.output_dim)
        if self.ntk_normalization:
            nn.init.normal_(final_layer.weight, mean=0, std=1.0 / torch.sqrt(torch.tensor(layer_input_dim, dtype=torch.float32)))
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class NTK:
    def __init__(self, x1, x2, model, progress_bar=True, eigenvalues=True):
        self.x1 = x1
        self.x2 = x2
        self.model = model
        self.progress_bar = progress_bar
        self.eigenvalues = eigenvalues

    def compute_ntk(self):
        kernel_matrix = self._compute_kernel_matrix(self.x1, self.x2)
        eigenvalues = None
        if self.eigenvalues and self.x1.shape[0] == self.x2.shape[0]:
            eigenvalues = torch.linalg.eigvals(kernel_matrix).real.to(dtype=torch.float32)
            eigenvalues = torch.abs(eigenvalues)
            eigenvalues = torch.sort(eigenvalues)[0]
        return (kernel_matrix, eigenvalues) if eigenvalues is not None else kernel_matrix

    def _compute_kernel_matrix(self, x1, x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        jacobians1 = []
        jacobians2 = []
        range_func = tqdm(range(n1), desc="Calculating Jacobians for x1") if self.progress_bar else range(n1)
        for i in range_func:
            x = x1[i].unsqueeze(0)
            jacobian = torch.autograd.functional.jacobian(lambda inp: self.model(inp).squeeze(), x, create_graph=True)
            jacobian_flat = torch.cat([j.view(-1) for j in jacobian], dim=0)
            jacobians1.append(jacobian_flat)
        jacobians1 = torch.stack(jacobians1)

        range_func = tqdm(range(n2), desc="Calculating Jacobians for x2") if self.progress_bar else range(n2)
        for i in range_func:
            x = x2[i].unsqueeze(0)
            jacobian = torch.autograd.functional.jacobian(lambda inp: self.model(inp).squeeze(), x, create_graph=True)
            jacobian_flat = torch.cat([j.view(-1) for j in jacobian], dim=0)
            jacobians2.append(jacobian_flat)
        jacobians2 = torch.stack(jacobians2)

        # Compute the NTK kernel matrix using torch.matmul for efficiency
        kernel_matrix = torch.matmul(jacobians1, jacobians2.T)
        return kernel_matrix

class Train:
    def __init__(self, x, t, model, opt=1, epochs=1000, learning_rate=0.001, progress_bar=True):
        self.x = x
        self.t = t
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.progress_bar = progress_bar

    def train_model(self):
        if self.opt == 1:  # Vanilla Gradient Descent
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Currently only Vanilla Gradient Descent (opt=1) is supported.")

        criterion = nn.MSELoss()
        losses = []
        range_func = tqdm(range(self.epochs), desc="Training Model") if self.progress_bar else range(self.epochs)
        for epoch in range_func:
            self.model.train()
            optimizer.zero_grad()
            y_pred = self.model(self.x)
            loss = criterion(y_pred, self.t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.16f}")

        return torch.tensor(losses), self.model(self.x)