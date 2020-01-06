"""Run sample experiments"""

import torch
from pytorch_lightning import Trainer
from rbfxor.modeling import BaseMLP
from rbfxor.functional import PiecewiseRBF

def run_exp(trainer, Ts, ep2s, hdims, rbf_at_0):
    x, y = len(Ts), len(ep2s)
    figscale = 6
    fig, axes = plt.subplots(x, y, figsize=(figscale*y, figscale*x))

    title = "T: {:0.3}, ep2: {}"

    for i, T in enumerate(Ts):
        for j, ep2 in enumerate(ep2s):
            act_func = PiecewiseRBF(T, rbf_at_0, ep2)
#             act_func = nn.Sigmoid()

            model = BaseMLP(hdims, act_func, final_act=None)
            trainer.fit(model)
            ax = axes[i, j]
            model.plot('train', ax)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_title(title.format(T, ep2))


if __name__ == "__main__":
    Ts = torch.linspace(0.01, 0.3, 5)
    ep2s = torch.linspace(0.1, 0.9, 5)
    hdims = [12]
    rbf_at_0 = 0.01
    trainer = Trainer()
    run_exp(trainer, Ts, ep2s, hdims, rbf_at_0)
