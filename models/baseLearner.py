import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from torch import optim
from types import SimpleNamespace

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 2

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

        


    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def tsne(self,showcenters=False,Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes=self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        vectors, y_true = self._extract_vectors(valloader)
        if showcenters:
            fc_weight=self._network.fc.proj.cpu().detach().numpy()[:tot_classes]
            print(fc_weight.shape)
            vectors=np.vstack([vectors,fc_weight])
        
        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(vectors)
        
        if showcenters:
            clssscenters=embedding[-tot_classes:,:]
            centerlabels=np.arange(tot_classes)
            embedding=embedding[:-tot_classes,:]
        scatter=plt.scatter(embedding[:,0],embedding[:,1],c=y_true,s=20,cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        if showcenters:
            plt.scatter(clssscenters[:,0],clssscenters[:,1],marker='*',s=50,c=centerlabels,cmap=plt.cm.get_cmap("tab20"),edgecolors='black')
        
        plt.savefig(str(self.args['model_name'])+str(tot_classes)+'tsne.pdf')
        plt.close()


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        init_cls = int(self.args.get("init_cls", 0))
        inc = int(self.args.get("increment", 0))
        if inc <= 0:
            # 全量微调场景：将分组步长退化为“单桶”
            # 优先用已知总类数；否则用标签上界+1；再不济用 1 兜底
            total_classes = getattr(self, "_total_classes", None)
            if total_classes is None:
                try:
                    total_classes = int(np.max(y_true)) + 1
                except Exception:
                    total_classes = 1
            inc = max(int(total_classes), 1)

        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, init_cls, inc)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def evaluate_full_dataset(self, loader=None):
        eval_loader = loader if loader is not None else getattr(self, "test_loader", None)
        if eval_loader is None:
            raise ValueError("No dataloader provided for evaluation.")
        y_pred, y_true = self._eval_cnn(eval_loader)
        return self._evaluate(y_pred, y_true)

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

        def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
            if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
                ori_classes = self._class_means.shape[0]
                assert ori_classes == self._known_classes
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._known_classes] = self._class_means
                self._class_means = new_class_means
                # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov[:self._known_classes] = self._class_covs
                self._class_covs = new_class_cov
            elif not check_diff:
                self._class_means = np.zeros((self._total_classes, self.feature_dim))
                # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

            for class_idx in range(self._known_classes, self._total_classes):

                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                try:
                    assert vectors.shape[0] > 1
                except AssertionError as e:
                    print("Size of the {}-th class is: {}, repeat it for twice.".format(class_idx, vectors.shape[0]))
                    vectors = np.tile(vectors, (2, 1))
                    print("Shape of vectors after repeating: {}".format(vectors.shape))

                # vectors = np.concatenate([vectors_aug, vectors])

                class_mean = np.mean(vectors, axis=0)
                # class_cov = np.cov(vectors.T)
                # try:
                #     class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-4
                # except UserWarning as e:
                #     logging.warning("Caught UserWarning: ", e)
               
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov


    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        policy: str = "constant",
        milestones=None,
        gamma: float = 1.0,
        T_max: int | None = None,
        eta_min: float | None = None,
    ):
        """Create a scheduler without hard-coding self.args inside.

        - policy: "multisteplr" | "cosine" | "constant"
        - milestones/gamma: used for MultiStepLR
        - T_max/eta_min: used for CosineAnnealingLR
        - constant: returns a no-op scheduler with `step()` method
        """
        policy = (policy).lower()
        if policy == "cosinelr" or policy == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=T_max, eta_min=(0.0 if eta_min is None else eta_min)
            )
        if policy == "multisteplr" or policy == "steplr":
            if milestones is None:
                milestones = []
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=float(gamma)
            )

        class _ConstScheduler:
            def step(self):
                return None

        # default constant LR
        return _ConstScheduler()

    
    def _resolve_optimizer_hyper(self, stage):
        if stage == "init":
            lr = float(self.args.get("init_lr", 0.1))
            momentum = float(self.args.get("init_momentum", 0.0))
            weight_decay = float(self.args.get("init_weight_decay", 0.0))
        elif stage == "update":
            lr = float(self.args.get("lrate", 0.1))
            momentum = float(self.args.get("momentum", 0.0))
            weight_decay = float(self.args.get("weight_decay", 0.0))
        elif stage == "full":  # full finetune
            lr = float(self.args.get("full_lr", 0.01))
            momentum = float(self.args.get("full_momentum", 0.0))
            weight_decay = float(self.args.get("full_weight_decay",0.0))
        return lr, momentum, weight_decay

    def _build_optimizer(self, params, stage):
        lr, momentum, weight_decay = self._resolve_optimizer_hyper(stage)
        print(f"_build_optimizer: {self._optimizer_type}")

        if self._optimizer_type == "sam":
            from optimer.optimer_sam import SAM

            print("init optimer-SAM")
            return SAM(
                params,
                optim.SGD,
                rho=self._sam_rho,
                adaptive=self._sam_adaptive,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        elif self._optimizer_type == "cflat":
            from optimer.c_flat import C_Flat

            base_opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            return C_Flat(
                params,
                base_optimizer=base_opt,
                model=self._network,
                cflat=True,
                rho=self._cflat_rho,
                lamb=self._cflat_lambda,
                adaptive=self._cflat_adaptive,
                perturb_eps=self._cflat_perturb_eps,
                grad_reduce=self._cflat_grad_reduce,
            )
        
        elif self._optimizer_type == "gam":
            # Gradient-Ascent Mask (GAM) optimizer; prefer finetune-initialized settings.
            from optimer.gam import GAM

            pre = getattr(self, "_gam_args", None)
            if pre is not None:
                # Use exactly what finetune (or child) prepared, with minimal fallback to attributes.
                adaptive = bool(getattr(self, "_gam_adaptive", getattr(pre, "adaptive", False)))
                grad_reduce = getattr(self, "_gam_grad_reduce", getattr(pre, "grad_reduce", "mean"))
                perturb_eps = float(getattr(self, "_gam_perturb_eps", getattr(pre, "perturb_eps", 1e-12)))
                grad_rho = float(getattr(pre, "grad_rho", getattr(self, "_gam_grad_rho", 0.2)))
                grad_norm_rho = float(getattr(pre, "grad_norm_rho", getattr(self, "_gam_grad_norm_rho", 0.2)))
                gam_args = SimpleNamespace(
                    grad_beta_1=float(getattr(pre, "grad_beta_1", getattr(self, "_gam_beta1", 1.0))),
                    grad_beta_2=float(getattr(pre, "grad_beta_2", getattr(self, "_gam_beta2", 1.0))),
                    grad_beta_3=float(getattr(pre, "grad_beta_3", getattr(self, "_gam_beta3", 1.0))),
                    grad_gamma=float(getattr(pre, "grad_gamma", getattr(self, "_gam_gamma", 0.1))),
                )
            else:
                # Fallback to attributes (set by child __init__), no args-based re-init here
                adaptive = bool(getattr(self, "_gam_adaptive", False))
                grad_reduce = getattr(self, "_gam_grad_reduce", "mean")
                perturb_eps = float(getattr(self, "_gam_perturb_eps", 1e-12))
                grad_rho = float(getattr(self, "_gam_grad_rho", 0.2))
                grad_norm_rho = float(getattr(self, "_gam_grad_norm_rho", 0.2))
                gam_args = SimpleNamespace(
                    grad_beta_1=float(getattr(self, "_gam_beta1", 1.0)),
                    grad_beta_2=float(getattr(self, "_gam_beta2", 1.0)),
                    grad_beta_3=float(getattr(self, "_gam_beta_3", getattr(self, "_gam_beta3", 1.0))),
                    grad_gamma=float(getattr(self, "_gam_gamma", 0.1)),
                )

            base_opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
            opt = GAM(
                params,
                base_optimizer=base_opt,
                model=self._network,
                grad_rho = grad_rho,
                grad_norm_rho = grad_norm_rho,
                adaptive=adaptive,
                perturb_eps=perturb_eps,
                args=gam_args,
                grad_reduce=str(grad_reduce),
            )
            # Use constant radii unless schedulers are attached inside GAM
            opt.grad_rho = torch.tensor(float(grad_rho)) if hasattr(torch, "tensor") else float(grad_rho)
            opt.grad_norm_rho = torch.tensor(float(grad_norm_rho)) if hasattr(torch, "tensor") else float(grad_norm_rho)
            return opt


        elif self._optimizer_type == "arwp":
            # Random Weight Perturbation optimizer (ARWP)
            from optimer.ARWP_cos import ARWP
            base_name = str(self.args.get("optimizer", "sgd")).lower()
            if base_name == "adam":
                base_opt_cls = optim.Adam
            elif base_name == "adamw":
                base_opt_cls = optim.AdamW
            else:
                base_opt_cls = optim.SGD
            return ARWP(
                params,
                base_opt_cls,
                std=float(getattr(self, "_rwp_std", self.args.get("rwp_std", 0.01))),
                eta=float(getattr(self, "_rwp_eta", self.args.get("rwp_eta", 1.0))),
                beta=float(getattr(self, "_rwp_beta", self.args.get("rwp_beta", 0.9))),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        
        elif self._optimizer_type == "rwp":
            opt_name = str(self.args.get("optimizer", "sgd")).lower()
            if opt_name == "adam":
                return optim.Adam(params, lr=lr, weight_decay=weight_decay)
            if opt_name == "adamw":
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        if self._optimizer_type == "adam":
            return optim.Adam(params)

        if self._optimizer_type == "adamw":
            return optim.AdamW(params)
        
        if self._optimizer_type == "sgd":
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


    
