import torch 
import torch.nn as nn
import torch.linalg
import torch.nn.functional as F



class DenseNTK(nn.Module):
    """
    Class for the layers of a FFNN
    I) Follows the torch.nn module (structure and syntax)
    II
    a) Initialize weights by ATS_NTK method (i.e. scaled by 1/sqrt(D))
    b) Correct the forward pass
    """

    def __init__(
        self,
        in_features, # input dim of layer
        out_features, # output dim of layer
        activation=torch.nn.Identity(), # no activation by auto
        bias=True, # bias added on auto
        initialization_=None): # 
        
        super(DenseNTK,self).__init__()
        # Initialize weights using ATS_NTK initialization
        if initialization_:
            self.weight = nn.Parameter(initialization_(in_features,out_features))
        else:
            # Default initialization with N(0, 1)/sqrt(in)
            self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias',None)

        self.activation = activation 
        
    def forward(self,x):
        weight_normalized = self.weight / (self.weight.size(1) ** 0.5)
        output = F.linear(x, weight_normalized, self.bias)
        output = self.activation(output)

        return output
    
class KernelMatrix():

    def __init__(self) -> None:
        pass

    def remove_last_bias(self,model,jacobian):
        """Removes last bias of model in jacobian"""
        # Get the size of the last bias parameter
        params = list(model.parameters())
        last_bias_size = params[-1].numel()  # Get the number of elements in the last bias parameter
        
        # Remove the columns corresponding to the last bias from the Jacobian
        jacobian = jacobian[:, :-last_bias_size]

        return jacobian

    def jacobian_torch(self, model, x, show_progress=False):
        if show_progress:
            import time
            from tqdm import tqdm
            start_time = time.time()
            print("\nComputing Jacobian with Torch...")

        # Ensure the model is in evaluation mode
        model.eval()

        # Forward pass through the model
        y = model(x)

        # Flatten output to make the computation easier if needed
        y_flat = y.view(-1)

        # Store the gradients for each parameter
        jacobian_list = []

        for i in tqdm(range(y_flat.size(0)), desc="Jacobian computation", disable=not show_progress):
            # Zero the gradients
            model.zero_grad()

            # Compute the gradient of the i-th output element with respect to all model parameters
            y_flat[i].backward(retain_graph=True)

            # Extract the gradient for each parameter and flatten it
            grads = []
            for param in model.parameters():
                grads.append(param.grad.view(-1))
            
            # Concatenate all gradients for the current output element
            jacobian_list.append(torch.cat(grads))

        # Stack all Jacobians together to form a matrix
        jacobian_matrix = torch.stack(jacobian_list).double()

        # Remove the contribution of the last bias parameter from the Jacobian
        jacobian_matrix = self.remove_last_bias(model,jacobian_matrix)

        if show_progress:
            print(f"\nTime taken: {time.time() - start_time} seconds")

    
        return jacobian_matrix,y