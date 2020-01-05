import torch
import matplotlib.pyplot as plt

def xor(x):
    """Given a 2D tensor x, return 1 if x[0] has different sign than x[1], otherwise return 0"""
    assert len(x.shape) == 2, "Expects 2 dimensional tensor!"
    assert x.shape[1] == 2, "Expect only 2 columns!"
    return torch.eq(torch.sign(x[:,0]) * torch.sign(x[:, 1]), -1)

def create_xor_dataset(N, seed=42):
    torch.manual_seed(seed)
    x = 2*torch.rand(N, 2) - 1
    y = xor(x).float().reshape((-1, 1))
    return x,y

def plot_xor_ds(x, y):
    plt.scatter(x[:,0].detach().numpy(), x[:,1].detach().numpy(), c=y.detach().numpy())
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Generated XOR dataset")
    plt.show()

class XORDataset():
    """Create XOR dataset"""
    def __init__(self, N, seed=42):
        self.X, self.Y = create_xor_dataset(N, seed)       

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]
    
    def plot(self):
        plot_xor_ds(self.X, self.Y.flatten())
