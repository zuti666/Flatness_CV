import logging
import numpy as np
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from backbone.net_ewclora import Net
from models_CL.baseLearner import BaseLearner
from models_LoRAbasedCL.baseLoRA import LoraBaseLearner
from utils.inc_net import IncrementalNet

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def print_trainable_params(model, show_shapes=True):
    total_params = 0
    trainable_params = 0

    print("Parameters to be updated:")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
            if show_shapes:
                print(f"[Trainable] {name:60s} {tuple(param.shape)} | {num_params}")
            else:
                print(f"[Trainable] {name:60s} | {num_params}")

    logging.info(f"Total params:     {total_params:,}")
    logging.info(f"Trainable params: {trainable_params:,}")
    logging.info(f"Trainable ratio:  {100 * trainable_params / total_params:.4f}%")


def check_params_consistency(model, optimizer):
    model_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    model_param_set = set(model_params.values())

    optim_param_set = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optim_param_set.add(p)

    only_in_model = model_param_set - optim_param_set
    only_in_optim = optim_param_set - model_param_set
    ok = (len(only_in_model) == 0 and len(only_in_optim) == 0)

    if ok:
        print("✅ Requires_grad parameters and optimizer parameters are consistent.")
    else:
        print("❌ WARNING: Inconsistency detected!")

    return ok




class EWCLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        
        self.topk = 1
        # Route through the framework backbone factory so this class follows
        # the same initialization path as other current methods.
        self._network = IncrementalNet(args, True)
        # Keep legacy field name/usage (`self.network`) for old training code.
        self.network = self._network.backbone if hasattr(self._network, "backbone") else self._network

        # ewclora
        self.gamma = args["gamma"]
        self.ewc_weight = args["lambda"]
        self.omega_W = []  # Importance matrix
        self.count_updates = 0
        
    def after_task(self):
        super().after_task()

        # Compute Fisher Information Matrix
        print("=== Update Importance Matrix ===")
        self.count_updates += 1
        fisher = FisherComputer(self.cur_task, self.network, self.train_loader, 
                                self.increment, F.cross_entropy, self.device)
        fisher_W = fisher.compute(max_batches=None)

        omega_W_bk = self.omega_W[:]
        self.omega_W = []

        new_a_params = filter(lambda p: getattr(p, '_is_new_a', False), self.network.parameters())
        new_b_params = filter(lambda p: getattr(p, '_is_new_b', False), self.network.parameters())
        for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
            if len(omega_W_bk) != 0:
                self.omega_W.append(self.gamma * omega_W_bk[idx] + fisher_W[idx])
            else:
                self.omega_W.append(fisher_W[idx])

        self.network.accumulate_and_reset_lora()

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        encoder_params = self.network.image_encoder.parameters()
        cls_params = [p for p in self.network.classifier_pool.parameters() if p.requires_grad==True]

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)
        
        encoder_params = {'params': encoder_params, 'lr': self.lrate, 'weight_decay': self.weight_decay}
        cls_params = {'params': cls_params, 'lr': self.fc_lrate, 'weight_decay': self.weight_decay}

        network_params = [encoder_params, cls_params]
        optimizer, scheduler = self.build_optimizer(network_params)
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)

        if len(self.multiple_gpus) > 1:
            self.network = self.network.module
        return

    def _train_function(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self.known_classes

                logits = self.network(inputs, use_new=True)['logits']
                loss = F.cross_entropy(logits, targets)

                # regularization loss
                if self.count_updates != 0:
                    new_a_params = filter(lambda p: getattr(p, '_is_new_a', False), self.network.parameters())
                    new_b_params = filter(lambda p: getattr(p, '_is_new_b', False), self.network.parameters())
                    ewc_loss = 0.
                    for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
                        delta_W = p_b @ p_a
                        ewc_term = self.omega_W[idx].type(torch.float32).to(self.device) * (delta_W ** 2)
                        ewc_loss += torch.sum(ewc_term)
                        
                    weighted_ewc_loss = self.ewc_weight/2. * ewc_loss
                    loss += weighted_ewc_loss
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            f"lora_new_A_k",
            f"lora_new_A_v",
            f"lora_new_B_k",
            f"lora_new_B_v",
        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))


class FisherComputer:
    def __init__(self, task_id, network, dataloader, increment, criterion, device=torch.device('cpu')):
        self.model = network.to(device)
        self.dataloader = dataloader
        self.increment = increment
        self.criterion = criterion
        self.device = device

        self.task_id = task_id
        self.fisher_W = []
        self._init_fisher_storage()

    def compute(self, max_batches=None):
        self.model.eval()
        num_samples = 0
        
        for i, (_, inputs, targets) in enumerate(tqdm(self.dataloader, desc="Computing Fisher")):
            if max_batches and i >= max_batches:
                break
            # Empirical Fisher
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            logits = self.model.forward(inputs, use_new=True, register_hook=True)['logits']
            targets = targets - self.task_id * self.increment
            loss = self.criterion(logits, targets)
            loss.backward()

            batch_size = inputs.size(0)
            num_samples += batch_size

            idx = 0
            for module in self.model.modules():
                if hasattr(module, 'delta_w_k_new_grad'):
                    grad_k = module.delta_w_k_new_grad
                    if grad_k is not None:
                        g2 = grad_k.detach().to(
                            device=self.fisher_W[idx].device, dtype=self.fisher_W[idx].dtype
                        ).pow(2)
                        self.fisher_W[idx] += g2 * batch_size
                    idx += 1
                if hasattr(module, 'delta_w_v_new_grad'):
                    grad_v = module.delta_w_v_new_grad
                    if grad_v is not None:
                        g2 = grad_v.detach().to(
                            device=self.fisher_W[idx].device, dtype=self.fisher_W[idx].dtype
                        ).pow(2)
                        self.fisher_W[idx] += g2 * batch_size
                    idx += 1                    
        if num_samples > 0:
            self.fisher_W = [fw / num_samples for fw in self.fisher_W]

        return self.fisher_W
    
    def _init_fisher_storage(self):
        for module in self.model.modules():
            if hasattr(module, 'lora_new_B_k') and hasattr(module, 'lora_new_A_k'):
                delta_w_k_new = module.lora_new_B_k.weight @ module.lora_new_A_k.weight
                self.fisher_W.append(torch.zeros_like(delta_w_k_new, device='cpu'))
            if hasattr(module, 'lora_new_B_v') and hasattr(module, 'lora_new_A_v'):
                delta_w_v_new = module.lora_new_B_v.weight @ module.lora_new_A_v.weight
                self.fisher_W.append(torch.zeros_like(delta_w_v_new, device='cpu'))


def _solve_sylvester_cg(B, A, GB, GA, eps=1e-6, tol=1e-6, maxiter=200, verbose=False):
    """
    (B B^T) G + G (A^T A) = GB A + B GA
    B: (m, r)
    A: (r, n)
    GB: (m, r)
    GA: (r, n)
    """
    m, n = B.shape[0], A.shape[1]
    R = GB @ A + B @ GA
    mn = m * n

    def matvec(vec):
        G = vec.view(m, n)
        MG = B @ (B.T @ G)
        GN = (G @ A.T) @ A
        out = MG + GN
        if eps != 0.0:
            out = out + eps * G
        return out.reshape(mn)

    b = R.reshape(mn)

    x_vec = torch.zeros_like(b)
    r_vec = b - matvec(x_vec)
    p = r_vec.clone()
    rsold = torch.dot(r_vec, r_vec)

    for k in range(maxiter):
        Ap = matvec(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-30)
        x_vec = x_vec + alpha * p
        r_vec = r_vec - alpha * Ap
        rsnew = torch.dot(r_vec, r_vec)
        if verbose:
            print(f"iter={k}, residual={rsnew.sqrt().item():.3e}")
        if torch.sqrt(rsnew) <= tol * torch.sqrt(torch.dot(b, b)):
            break
        beta = rsnew / (rsold + 1e-30)
        p = r_vec + beta * p
        rsold = rsnew

    return x_vec.view(m, n)


from models_LoRAbasedCL.ewclora import Learner as _IntegratedEWCLoRA


class Learner(_IntegratedEWCLoRA):
    """Compatibility entrypoint using the framework-integrated EWCLoRA runner."""

    def __init__(self, args):
        # Avoid constructing Net(args) twice in parent __init__;
        # keep the same hyperparameter semantics as integrated EWCLoRA.
        LoraBaseLearner.__init__(self, args)
        self.topk = 1
        self._network = IncrementalNet(args, True)
        self._optimizer_type = args.get("optimizer_type", "sgd").lower()
        if self._optimizer_type == "sam":
            self._sam_rho = float(args.get("sam_rho", 0.05))
            self._sam_adaptive = bool(args.get("sam_adaptive", False))
        elif self._optimizer_type == "cflat":
            self._cflat_rho = float(args.get("cflat_rho", 0.2))
            self._cflat_lambda = float(args.get("cflat_lambda", 0.2))
            self._cflat_adaptive = bool(args.get("cflat_adaptive", False))
            self._cflat_perturb_eps = float(args.get("cflat_perturb_eps", 1e-12))
            self._cflat_grad_reduce = args.get("cflat_grad_reduce", "mean")
        # elif self._optimizer_type == "gam":
        #     # Core GAM flags
        #     self._gam_adaptive = bool(args.get("gam_adaptive", False))
        #     self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
        #     self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
        #     # Two perturbation radii (ρ for loss-grad step; ρ' for norm-ascent step)
        #     self._gam_grad_rho = float(args.get("gam_grad_rho", 0.2))
        #     self._gam_grad_norm_rho = float(args.get("gam_grad_norm_rho", 0.2))
        #     # Gradient decomposition weights used in optimer.gam.GAM.gradient_decompose
        #     self._gam_beta1 = float(args.get("gam_grad_beta_1", 1.0))
        #     self._gam_beta2 = float(args.get("gam_grad_beta_2", 1.0))
        #     self._gam_beta3 = float(args.get("gam_grad_beta_3", 1.0))
        #     self._gam_gamma = float(args.get("gam_grad_gamma", 0.1))

        #     # Pack into a namespace for downstream optimizers (BaseLearner can read self._gam_args)
        #     self._gam_args = SimpleNamespace(
        #         # decomposition weights
        #         grad_beta_1=self._gam_beta1,
        #         grad_beta_2=self._gam_beta2,
        #         grad_beta_3=self._gam_beta3,
        #         grad_gamma=self._gam_gamma,
        #         # radii (optional convenience)
        #         grad_rho=self._gam_grad_rho,
        #         grad_norm_rho=self._gam_grad_norm_rho,
        #         # misc flags (optional convenience)
        #         adaptive=self._gam_adaptive,
        #         perturb_eps=self._gam_perturb_eps,
        #         grad_reduce=str(self._gam_grad_reduce),
        #     )
        # elif self._optimizer_type == "gam":
        #     self._gam_adaptive = bool(args.get("gam_adaptive", False))
        #     self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
        #     self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
        #     self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.05)))
        #     self._gam_grad_norm_rho = float(
        #         args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.05))
        #     )
        #     self._gam_beta1 = float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 0.5)))
        #     self._gam_beta2 = float(args.get("gam_grad_beta_2", args.get("grad_beta_2", 0.5)))
        #     self._gam_beta3 = float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 0.5)))
        #     self._gam_gamma = float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.5)))
        #     self._gam_args = SimpleNamespace(
        #         grad_beta_1=self._gam_beta1,
        #         grad_beta_2=self._gam_beta2,
        #         grad_beta_3=self._gam_beta3,
        #         grad_gamma=self._gam_gamma,
        #         grad_rho=self._gam_grad_rho,
        #         grad_norm_rho=self._gam_grad_norm_rho,
        #         adaptive=self._gam_adaptive,
        #         perturb_eps=self._gam_perturb_eps,
        #         grad_reduce=str(self._gam_grad_reduce),
        #     )
        elif self._optimizer_type == "gam":
            self._gam_adaptive = bool(args.get("gam_adaptive", False))
            self._gam_grad_reduce = args.get("gam_grad_reduce", "mean")
            self._gam_perturb_eps = float(args.get("gam_perturb_eps", 1e-12))
            self._gam_grad_rho = float(args.get("gam_grad_rho", args.get("grad_rho", 0.02)))
            self._gam_grad_norm_rho = float(
                args.get("gam_grad_norm_rho", args.get("grad_norm_rho", 0.2))
            )
            self._gam_beta1 = float(args.get("gam_grad_beta_1", args.get("grad_beta_1", 1)))
            self._gam_beta2 = float(args.get("gam_grad_beta_2", args.get("grad_beta_2", -1)))
            self._gam_beta3 = float(args.get("gam_grad_beta_3", args.get("grad_beta_3", 1)))
            self._gam_gamma = float(args.get("gam_grad_gamma", args.get("grad_gamma", 0.03)))
            self._gam_args = SimpleNamespace(
                grad_beta_1=self._gam_beta1,
                grad_beta_2=self._gam_beta2,
                grad_beta_3=self._gam_beta3,
                grad_gamma=self._gam_gamma,
                grad_rho=self._gam_grad_rho,
                grad_norm_rho=self._gam_grad_norm_rho,
                adaptive=self._gam_adaptive,
                perturb_eps=self._gam_perturb_eps,
                grad_reduce=str(self._gam_grad_reduce),
            )
        self._gamma = float(args.get("gamma", args.get("ewc_gamma", 1.0)))
        self._ewc_weight = float(args.get("lambda", args.get("ewc_lambda", 20.0)))
        fisher_max = args.get("ewc_max_batches", None)
        self._fisher_max_batches = None if fisher_max is None else int(fisher_max)
        self._omega_w = []
        self._count_updates = 0
