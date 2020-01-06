import numpy as np

def rbf(x, eps=1):
    """Calculate the 1D gaussian RBF at x given an epsilon"""
    return np.exp(-(eps * x)**2)

def drbf(x, eps=1):
    """Calculate the derivative of a 1D gaussian RBF at x given an epsilon"""
    return (-2 * eps**2 * x) * np.exp(-(eps*x)**2)

def T2eps(T, s=0.001):
    """Calculate the epsilon needed such that rbf(T, eps) = s
    
    's' is a small number approximately 0 and T is the value at w"""
    return 1/(2*T) * np.log(1/s)

def piecewise_rbf(x, T=0.2, ep2=0.01, rbf_at_0=0.001):
    """ Create a piecewise RBF activation function
    
    Main parameters:
    - T : The threshold for the activation function
    - ep2: The epsilon hyperparameter for the second half of the piecewise RBF
    - rbf_at_0: The value the first half of the piecewise RBF crosses at x=0.
    """
    assert (rbf_at_0 > 0), "RBF must be between (0 and 1)"
    assert (T > 0), "Activation function assumes positive T value"
    
    if x < T:
        ep1 = T2eps(T, s=rbf_at_0)
        return rbf(x-2, ep1)
    
    return rbf(x-T, ep2)
