import torch
from torch.optim import AdamW
import torch.nn as nn
from typing import Iterable
import logging
import os
import sys

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.DEBUG,
            handlers=[logging.StreamHandler(sys.stdout)],
        )


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Initialize the SAM (Sharpness-Aware Minimization) optimizer

        Args:
            params: List of training parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.SGD or torch.optim.Adam)
            rho: Hyperparameter controlling the perturbation magnitude
            adaptive: Whether to use parameter-size dependent adaptive perturbation
            **kwargs: Parameters passed to the base optimizer
        """

        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # Initialize default parameters
        defaults = dict(rho=rho, adaptive=adaptive,** kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups  # Bind parameter groups (ensure consistency)



    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: On the original parameter point w, calculate gradients and perform an "uphill" perturbation
        (to find the sharp position)

        Args:
            zero_grad: Whether to clear gradients at the end
        """
        grad_norm = self._grad_norm()  # Calculate overall L2 gradient norm
        logger.debug(f"first_step - Overall L2 gradient norm: {grad_norm}")

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # Get perturbation scaling factor based on rho and norm
            logger.debug(f"rho: {group['rho']}, grad_norm: {grad_norm}, scale: {scale}")

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()  # Save original parameters for restoration

                # e(w): Calculate perturbation direction
                rescale = (torch.pow(p, 2) if group["adaptive"] else 1.0)
                e_w = rescale * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"

        if zero_grad:
            logger.debug("first_step - Clearing gradients")
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: Restore from original parameter point w, then perform actual update using gradients
        calculated at w + e(w)

        Args:
            zero_grad: Whether to clear gradients at the end
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"

        logger.debug("second_step - Executing base optimizer step operation")
        self.base_optimizer.step()  # Perform the actual "sharpness-aware" update

        if zero_grad:
            logger.debug("second_step - Clearing gradients")
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Core step function of SAM. Must pass a closure function before calling, completing the full process:
            - w → w + e(w)
            - Calculate gradients at w + e(w)
            - Update parameters at w

        Args:
            closure: Closure function that encapsulates the complete forward + backward propagation process
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # Re-enable grad mode to ensure closure can perform backpropagation
        closure = torch.enable_grad()(closure)  # The closure should perform a full forward-backward pass

        self.first_step(zero_grad=True)   # First step: Add perturbation and clear gradients
        closure()  # Second step: Calculate gradients at w + e(w)
        self.second_step()  # Third step: Restore w and perform update with gradients from previous step

    def _grad_norm(self):
        """
        Calculate L2 norm of current parameters, used to normalize perturbation direction
        """
        shared_device = self.param_groups[0]["params"][0].device  # Put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def disable_running_stats(model):
    """
    Call before first_step to disable BatchNorm statistics tracking
    """
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    """
    Call after restoring model (after second_step) to restore BatchNorm momentum
    """
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)



class SAMAdamW(AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, rho=0.05):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self, closure) -> torch.Tensor:
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']

            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads
            torch._foreach_mul_(epsilon, rho / grad_norm)

            torch._foreach_add_(params_with_grads, epsilon)
            closure()
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss