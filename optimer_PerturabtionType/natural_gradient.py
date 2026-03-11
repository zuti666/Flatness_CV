import torch
from torch.optim import Optimizer


class NaturalGradient(Optimizer):
    """Diagonal natural gradient optimizer using an EMA Fisher estimate.

    Fisher is accumulated as an exponential moving average of squared gradients,
    and parameters are updated with a simple preconditioned step:
        theta <- theta - lr * grad / (fisher + damping + eps)
    """

    def __init__(self, params, lr=1e-2, damping=1e-3, ema_decay=0.95, eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"Invalid ema_decay: {ema_decay}")
        defaults = dict(lr=lr, damping=damping, ema_decay=ema_decay, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            ema_decay = group["ema_decay"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if "fisher" not in state:
                    state["fisher"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                fisher = state["fisher"]

                fisher.mul_(ema_decay).addcmul_(grad, grad, value=(1.0 - ema_decay))
                precond = 1.0 / (fisher + damping + eps)
                p.addcmul_(grad, precond, value=-lr)

        return loss
