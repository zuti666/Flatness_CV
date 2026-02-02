import torch
import numpy as np


class ARWP(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, std=0.01, eta=1, beta=0.9, **kwargs):
        assert std >= 0.0, f"Invalid std, should be non-negative: {std}"

        defaults = dict(std=std, **kwargs)
        super(ARWP, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.std = std
        self.eta = eta
        self.beta = beta
        print ('ARWP std:', self.std, 'eta:', self.eta, 'beta:', self.beta)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        for group in self.param_groups:

            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                sh = p.data.shape
                sh_mul = int(np.prod(sh[1:]))

                fisher = None
                pd = p.data 
                if "old_g" in self.state[p]:
                    fisher = self.state[p]["old_g"].view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)

                if len(p.data.shape) > 1:
                    e_w = pd.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                    e_w = torch.normal(0, (self.std + 1e-16) * e_w).to(p)
                else:
                    e_w = torch.empty_like(pd).to(p)
                    e_w.normal_(0, self.std * (pd.view(-1).norm().item() + 1e-16))
                    
                if fisher is not None:
                    e_w /= torch.sqrt(1 + self.eta * fisher)

                p.add_(e_w)  # add weight noise

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if "old_g" not in self.state[p]:
                    self.state[p]["old_g"] = p.grad.clone() ** 2
                else:
                    self.state[p]["old_g"] = self.state[p]["old_g"] * self.beta + p.grad.clone() ** 2

        self.base_optimizer.step()  # do the actual update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "RWP requires closure, but it was not provided"
        # the closure should do a full forward-backward pass and return (outputs, loss_value)
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        outputs, loss_value = closure()
        self.second_step()
        return outputs, loss_value

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups  
