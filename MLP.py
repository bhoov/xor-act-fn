import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from dataset import XORDataset
from pytorch_lightning import Trainer
    
def create_mod_list(init_dim, out_dim, hdims, activation, final_act=nn.Sigmoid()):
    if len(hdims) == 0:
        return nn.Sequential(*[nn.Linear(init_dim, out_dim), final_act])
    
    mod_list = [nn.Linear(init_dim, hdims[0]), activation]
    
    for i in range(len(hdims) - 1):
        ind = i + 1
        mod_list += [nn.Linear(hdims[i], hdims[ind]), activation]
        
    mod_list += [nn.Linear(hdims[-1], out_dim), final_act]
    return nn.Sequential(*mod_list)

def plot_model(model, X, y):
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
        return DataLoader(XORDataset(1000, self.seed), batch_size=32, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return DataLoader(XORDataset(200, self.seed), batch_size=64, shuffle=False)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        # can also return a list of test dataloaders
        return DataLoader(XORDataset(100, self.seed), batch_size=64, shuffle=False)

if __name__ == "__main__":
    ds_train = XORDataset(1000, 33)
    ds_test = XORDataset(100, 32)

    model = BaseMLP([100], nn.ReLU(), nn.Sigmoid())
    trainer = Trainer()
    trainer.fit(model)
    labels = plot_model(model, ds_test.X, ds_test.Y.flatten())
    
