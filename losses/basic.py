import torch.nn.functional as F

def nll_loss(im, tar):
    return F.nll_loss(im, tar)