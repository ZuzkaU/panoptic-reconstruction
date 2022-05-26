import torch

def reconstruction_loss(input, target, beta=1. / 9, size_average=True):
    """
    TODO - this is copied from smooth_l1_loss
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
