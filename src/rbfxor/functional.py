import torch
from torch.autograd import Function
import torch.nn as nn
import matplotlib.pyplot as plt

def gaussian_rbf(x, ep):
    """Calculate the gaussian RBF of a tensor 'x', parameterized by 'ep'"""
    return torch.exp(-(ep * x)**2)

def gaussian_drbf(x, ep):
    """Calculate the derivative of the gaussian RBF of a tensor 'x', parameterized by 'ep'"""
    return (-2 * ep**2 * x) * torch.exp(-(ep * x)**2)

def piecewise_rbf_forward(input, T, ep1, ep2):
    """Compute the piecewise activation function in the forward direction

    Parameters:
    - T: Threshold at which the RBFs behave differently
    - ep1: The epsilon parameter for the first half of the piecewise function. Assumed to be calculated by
        'rbf_at_0' in the activation function module
    - ep2: The epsilon parameter of the 2nd half of hte piecewise function.
    """

    output = input.clone()
    below_T_inds = input < T
    above_T_inds = input >= T

    output[below_T_inds] = gaussian_rbf(input-T, ep1)[below_T_inds]
    output[above_T_inds] = gaussian_rbf(input-T, ep2)[above_T_inds]
    return output

def piecewise_rbf_backward(input, T, ep1, ep2):
    """Calculate the backward pass for the rbf kernel.

    Returns the gradient for each of the input values
    """
    output = input.clone()
    below_T_inds = input < T
    above_T_inds = input >= T

    output[below_T_inds] = gaussian_drbf(input-T, ep1)[below_T_inds]
    output[above_T_inds] = gaussian_drbf(input-T, ep2)[above_T_inds]
    return output

class piecewise_rbf(Function):
    """Define the forward pass and backward pass with context for Pytorch"""
    @staticmethod
    def forward(ctx, input, T, ep1, ep2):
        ctx.save_for_backward(input, T, ep1, ep2)
        return piecewise_rbf_forward(input, T, ep1, ep2)

    @staticmethod
    def backward(ctx, grad_output):
        input, T, ep1, ep2 = ctx.saved_variables
        return piecewise_rbf_backward(input, T, ep1, ep2) * grad_output, None, None, None

class PiecewiseRBF(nn.Module):
    """Module representing the piecewise RBF activation function"""
    def __init__(self, T=0.1, rbf_at_0=0.001, ep2=0.8):
        super().__init__()
        self.T = torch.tensor(T)
        self.rbf_at_0 = torch.tensor(rbf_at_0)
        self.ep2 = torch.tensor(ep2)

    @property
    def ep1(self):
        """Calculate the ep1 from the 'rbf_at_0' parameter"""
        return 1./(2.*self.T) * torch.log(1/self.rbf_at_0)

    def forward(self, x):
        return piecewise_rbf.apply(x, self.T, self.ep1, self.ep2)

    def plot(self, xmin=-1, xmax=5):
        """Show a 1D plot of the activation function"""
        x = torch.linspace(xmin, xmax, 1000)
        y = self(x)
        plt.plot(x.detach().numpy(), y.detach().numpy())
        def rounder(s): return round(s, 3)
        show_params = "\n".join(["T = " + str(rounder(self.T.item())), "ep1 = " + str(rounder(self.ep1.item())), "ep2 = " + str(rounder(self.ep2.item()))])
        plt.legend([show_params], loc="best")
        plt.show()
