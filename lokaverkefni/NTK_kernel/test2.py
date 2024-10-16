import torch 
import torch.nn 
import numpy as np
















if __name__ == "__main__":
    x1 = [1,-1]
    x2 = [2,0]
    x3 = [3,-3]
    x4 = [4,-4]

    x = torch.tensor([x1,x2,x3,x4])
    t = torch.exp(x)
    print(x.ndim)