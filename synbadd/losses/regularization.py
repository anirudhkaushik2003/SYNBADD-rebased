"""Regularization losses for deformation fields."""

import torch


def gradient_loss(flow):
    """L1 spatial gradient loss for a flow field."""
    dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy) + torch.mean(dz)


def smooth_loss_l2(y_pred, penalty="l2", loss_mult=1):
    """Smoothness penalty on a displacement/velocity field."""
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

    if penalty == "l2":
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0

    if loss_mult is not None:
        grad *= loss_mult
    return grad


def r1_reg(d_out, x_in):
    """R1 zero-centered gradient penalty for real images."""
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    return 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
