import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from .dataset import XORDataset
from .functional import PiecewiseRBF
from typing import List

def create_mod_list(init_dim: int, out_dim: int, hdims: List[int], activation, final_act):
    """Create a simplified MLP specifying certain parameters.
    
    Parameters:
    - init_dim: Dimension of input to network
    - out_dim: Dimension of output from network
    - hdims: List of hidden dimensions to create a sequential model. If empty, no hidden layers are created.
    - activation: Activation function to go between each of the hidden layers.
    - final_activation: (Can be None). Activation function to place at the output of the network
    """
    if len(hdims) == 0:
        return nn.Sequential(*[nn.Linear(init_dim, out_dim), final_act])
    
    mod_list = [nn.Linear(init_dim, hdims[0]), activation]
    
    for i in range(len(hdims) - 1):
        ind = i + 1
        mod_list += [nn.Linear(hdims[i], hdims[ind]), activation]
        
    mod_list += [nn.Linear(hdims[-1], out_dim)]
        
    if final_act is not None: mod_list += [final_act]
    return nn.Sequential(*mod_list)

def plot_model(model, X, y):
    """Plot decision regions of the model atop a test dataset X and y"""
    cmap = plt.get_cmap("Paired")
    
    xs = np.linspace(-1.1, 1.1, 100)
    ys = np.linspace(-1.1, 1.1, 100)
    xx, yy = np.meshgrid(xs, ys)
    input = torch.tensor([xx.ravel(), yy.ravel()]).T.float()
    z = model.forward(input).reshape(xx.shape).detach()
    z[z < 0.5] = 0
    z[z >= 0.5] = 1
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    
    return fig, ax

    
class BaseMLP(pl.LightningModule):
    def __init__(self, hdims, activation, final_act=nn.Sigmoid(), seed=42):
        """
        Parameters:
        ===========
        - hdims (List[int]): Hidden dimensions for each layer
        """
        super().__init__()
        
        self.layers = create_mod_list(2, 1, hdims, activation, final_act)
        self.loss = nn.MSELoss()
        self.batch_size = 32
        self.seed = seed

    def forward(self, x):
        x = self.layers(x)
        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': self.loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.loss(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.loss(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
#         return torch.optim.SGD(self.parameters(), lr=0.002, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(XORDataset(500, self.seed), batch_size=32, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return DataLoader(XORDataset(200, self.seed+1), batch_size=64, shuffle=False)
    
    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        # can also return a list of test dataloaders
        return DataLoader(XORDataset(200, self.seed+2), batch_size=64, shuffle=False)
    
    def plot(self, ds_type='test'):
        """Plot the decision regions of the model. 
        
        Parameters:
        - ds_type: {'train', 'test', 'val'} - Which dataset to plot on top of the regions. Default 'test'
        """
        ds_map = {
            'train': self.train_dataloader().dataset,
            'test': self.test_dataloader()[0].dataset,
            'val': self.val_dataloader()[0].dataset
        }
        ds = ds_map[ds_type]
        
        plot_model(self, ds.X, ds.Y.flatten())
