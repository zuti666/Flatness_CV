import torch
from torch.utils.data import DataLoader


def _ensure_eval_mode(module):
    was_training = module.training
    module.eval()
    return was_training


def _restore_mode(module, was_training):
    if was_training:
        module.train()


def _extract_features(network, loader, device, max_batches=None):
    features = []
    labels = []

    module = network.module if isinstance(network, torch.nn.DataParallel) else network
    was_training = _ensure_eval_mode(module)

    with torch.no_grad():
        for batch_idx, (_, inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            outputs = module.extract_vector(inputs)
            features.append(outputs.detach().cpu())
            labels.append(targets.detach().cpu())

    _restore_mode(module, was_training)

    if not features:
        return None, None

    return torch.cat(features, dim=0).float(), torch.cat(labels, dim=0).long()


def _fit_ridge_classifier(features, labels, num_classes, l2_reg=1e-3):
    device = features.device
    n_samples, feat_dim = features.shape
    eye = torch.eye(feat_dim, device=device)

    one_hot = torch.zeros(n_samples, num_classes, device=device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    XtX = features.T @ features + l2_reg * eye
    XtY = features.T @ one_hot

    try:
        weights = torch.linalg.solve(XtX, XtY)
    except RuntimeError:
        weights = torch.linalg.lstsq(XtX, XtY).solution

    return weights


def evaluate_linear_probe(
    network,
    train_loader,
    test_loader,
    class_offset,
    num_classes,
    device,
    l2_reg=1e-3,
    max_train_batches=None,
    max_test_batches=None,
):
    train_feats, train_lbls = _extract_features(
        network, train_loader, device, max_batches=max_train_batches
    )
    test_feats, test_lbls = _extract_features(
        network, test_loader, device, max_batches=max_test_batches
    )

    if train_feats is None or test_feats is None:
        return float("nan")

    train_lbls_local = (train_lbls - class_offset).to(torch.long)
    test_lbls_local = (test_lbls - class_offset).to(torch.long)

    weights = _fit_ridge_classifier(train_feats, train_lbls_local, num_classes, l2_reg=l2_reg)

    logits = test_feats @ weights.cpu()
    preds = torch.argmax(logits, dim=1)
    correct = (preds == test_lbls_local).float().mean().item()

    return correct * 100.0
