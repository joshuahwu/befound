"""
GPU-native replacements for scrubvae.eval.metrics's log_class_rand_cv/
qda_rand_cv -- same windowed-downsample + K-fold cross-validated
decodability check (how accurately a held-out classifier can recover a
label from a latent z), but staying in torch end-to-end instead of
sklearn/numpy, since befound's loaders are far larger than what those
sklearn-based helpers were built for.
"""

import torch
import torch.nn.functional as F


def _windowed_downsample(z: torch.Tensor, y: torch.Tensor, window: int):
    """Every `window`-th row -- consecutive windows overlap heavily, so
    this avoids near-duplicate rows landing in both the train and test
    splits of a fold (matches scrubvae.eval.metrics.rand_cv's downsample,
    restricted to its only actually-used shift, start_i=0)."""
    return z[::window], y[::window]


def _kfold_indices(n: int, folds: int, seed: int = 100, device=None):
    """`folds` roughly-equal, shuffled splits of range(n); each is used
    once as the held-out test set, the rest as train -- matches
    sklearn.model_selection.KFold(shuffle=True, random_state=100)'s split
    structure closely enough for this purpose."""
    perm = torch.randperm(n, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    chunks = torch.chunk(perm, folds)
    for i in range(folds):
        test_i = chunks[i]
        train_i = torch.cat([chunks[j] for j in range(folds) if j != i])
        yield train_i, test_i


def linear_regression_rand_cv_torch(
    z: torch.Tensor,
    y: torch.Tensor,
    window: int = 51,
    folds: int = 5,
    reg: float = 1e-6,
):
    """
    GPU-native stand-in for scrubvae.eval.metrics.linear_rand_cv.

    Closed-form ridge least squares (normal equations) -- no training loop:
    fits on each fold's train split, scores held-out R^2 on its test split.
    Meant for a continuous target (e.g. the fps/rotation augmentation's own
    sampled value), unlike log_class_rand_cv_torch/qda_rand_cv_torch/
    lda_rand_cv_torch below, which are for a categorical one.

    Parameters
    ----------
    z : torch.Tensor, (N, D)
    y : torch.Tensor, (N,) or (N, Y)

    Returns
    -------
    List[float], one held-out R^2 per fold (mean across output columns if
    y has more than one, sklearn's multioutput="uniform_average")
    """
    if y.dim() == 1:
        y = y[:, None]
    y = y.to(z.dtype)

    z_ds, y_ds = _windowed_downsample(z, y, window)
    device = z_ds.device

    r2s = []
    for train_i, test_i in _kfold_indices(len(z_ds), folds, device=device):
        z_train, y_train = z_ds[train_i], y_ds[train_i]
        z_test, y_test = z_ds[test_i], y_ds[test_i]

        ones_train = torch.ones(z_train.shape[0], 1, device=device, dtype=z.dtype)
        z_train_aug = torch.cat([z_train, ones_train], dim=1)
        eye = torch.eye(z_train_aug.shape[1], device=device, dtype=z.dtype)
        eye[-1, -1] = 0  # don't regularize the bias term
        w = torch.linalg.solve(
            z_train_aug.T @ z_train_aug + reg * eye, z_train_aug.T @ y_train
        )

        ones_test = torch.ones(z_test.shape[0], 1, device=device, dtype=z.dtype)
        z_test_aug = torch.cat([z_test, ones_test], dim=1)
        y_pred = z_test_aug @ w

        ss_res = ((y_test - y_pred) ** 2).sum(dim=0)
        ss_tot = ((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        r2_per_output = 1 - ss_res / ss_tot.clamp_min(1e-12)
        r2s.append(r2_per_output.mean().item())

    return r2s


def log_class_rand_cv_torch(
    z: torch.Tensor,
    y: torch.Tensor,
    window: int = 51,
    folds: int = 5,
    n_steps: int = 300,
    lr: float = 0.05,
):
    """
    GPU-native stand-in for scrubvae.eval.metrics.log_class_rand_cv.

    Fits a small linear softmax classifier (torch.nn.Linear + Adam) on
    each fold's train split, scores held-out accuracy on its test split.

    Parameters
    ----------
    z : torch.Tensor, (N, D)
    y : torch.Tensor, (N,), integer class labels in [0, n_classes)

    Returns
    -------
    List[float], one held-out accuracy per fold
    """
    z_ds, y_ds = _windowed_downsample(z, y, window)
    y_ds = y_ds.long()
    device = z_ds.device
    n_classes = int(y_ds.max().item()) + 1

    accs = []
    for train_i, test_i in _kfold_indices(len(z_ds), folds, device=device):
        z_train, y_train = z_ds[train_i], y_ds[train_i]
        z_test, y_test = z_ds[test_i], y_ds[test_i]

        clf = torch.nn.Linear(z_train.shape[-1], n_classes, device=device)
        optimizer = torch.optim.AdamW(clf.parameters(), lr=lr)
        clf.train()
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = F.cross_entropy(clf(z_train), y_train)
            loss.backward()
            optimizer.step()

        clf.eval()
        with torch.no_grad():
            y_pred = clf(z_test).argmax(dim=-1)
            accs.append((y_pred == y_test).float().mean().item())

    return accs


def lda_rand_cv_torch(
    z: torch.Tensor,
    y: torch.Tensor,
    window: int = 51,
    folds: int = 5,
    reg: float = 1e-3,
):
    """
    GPU-native stand-in for scrubvae.eval.metrics.lda_rand_cv.

    Closed-form linear discriminant analysis -- like qda_rand_cv_torch, but
    with a single pooled (within-class) covariance shared across every
    class, rather than each class fitting its own. That shared covariance
    makes every class's quadratic term identical, so it drops out of the
    argmax entirely and the decision boundaries between classes are linear
    (hence "linear" discriminant analysis).

    Parameters
    ----------
    z : torch.Tensor, (N, D)
    y : torch.Tensor, (N,), integer class labels in [0, n_classes)

    Returns
    -------
    List[float], one held-out accuracy per fold
    """
    z_ds, y_ds = _windowed_downsample(z, y, window)
    y_ds = y_ds.long()
    device = z_ds.device
    n_classes = int(y_ds.max().item()) + 1
    D = z_ds.shape[-1]
    eye = torch.eye(D, device=device)

    accs = []
    for train_i, test_i in _kfold_indices(len(z_ds), folds, device=device):
        z_train, y_train = z_ds[train_i], y_ds[train_i]
        z_test, y_test = z_ds[test_i], y_ds[test_i]
        n_train = len(z_train)

        means, log_priors = [], []
        scatter = torch.zeros(D, D, device=device)
        for c in range(n_classes):
            z_c = z_train[y_train == c]
            n_c = len(z_c)
            mu_c = z_c.mean(dim=0) if n_c > 0 else torch.zeros(D, device=device)
            centered = z_c - mu_c
            scatter = scatter + centered.T @ centered

            means.append(mu_c)
            # log(0) for an empty class -- never the argmax, correctly
            # excludes it rather than crashing.
            log_priors.append(torch.log(torch.tensor(n_c / n_train, device=device)))

        means = torch.stack(means)  # (C, D)
        log_priors = torch.stack(log_priors)  # (C,)

        # Pooled within-class covariance (one degree of freedom used per
        # class, for its own mean), shared by every class.
        cov = scatter / max(n_train - n_classes, 1) + reg * eye
        precision = torch.linalg.inv(cov)

        diff = z_test[:, None, :] - means[None, :, :]  # (N, C, D)
        mahalanobis = torch.einsum("ncd,de,nce->nc", diff, precision, diff)
        scores = -0.5 * mahalanobis + log_priors[None, :]
        y_pred = scores.argmax(dim=-1)
        accs.append((y_pred == y_test).float().mean().item())

    return accs


def qda_rand_cv_torch(
    z: torch.Tensor,
    y: torch.Tensor,
    window: int = 51,
    folds: int = 5,
    reg: float = 1e-3,
):
    """
    GPU-native stand-in for scrubvae.eval.metrics.qda_rand_cv.

    Closed-form quadratic discriminant analysis -- no training loop: fits
    a per-class Gaussian (mean + ridge-regularized covariance) on each
    fold's train split, scores held-out accuracy by argmax class
    log-posterior on its test split. Same math as
    scrubvae.model.disentangle.QuadraticDiscriminantFilter's two streaming
    decoders, just computed once in closed form rather than as a moving
    average.

    Parameters
    ----------
    z : torch.Tensor, (N, D)
    y : torch.Tensor, (N,), integer class labels in [0, n_classes)

    Returns
    -------
    List[float], one held-out accuracy per fold
    """
    z_ds, y_ds = _windowed_downsample(z, y, window)
    y_ds = y_ds.long()
    device = z_ds.device
    n_classes = int(y_ds.max().item()) + 1
    D = z_ds.shape[-1]
    eye = torch.eye(D, device=device)

    accs = []
    for train_i, test_i in _kfold_indices(len(z_ds), folds, device=device):
        z_train, y_train = z_ds[train_i], y_ds[train_i]
        z_test, y_test = z_ds[test_i], y_ds[test_i]
        n_train = len(z_train)

        means, precisions, logdets, log_priors = [], [], [], []
        for c in range(n_classes):
            z_c = z_train[y_train == c]
            n_c = len(z_c)
            mu_c = z_c.mean(dim=0) if n_c > 0 else torch.zeros(D, device=device)
            centered = z_c - mu_c
            cov_c = (centered.T @ centered) / max(n_c - 1, 1) + reg * eye
            sign, logdet_c = torch.linalg.slogdet(cov_c)

            means.append(mu_c)
            precisions.append(torch.linalg.inv(cov_c))
            logdets.append(logdet_c)
            # log(0) for an empty class -- never the argmax, correctly
            # excludes it rather than crashing.
            log_priors.append(torch.log(torch.tensor(n_c / n_train, device=device)))

        means = torch.stack(means)  # (C, D)
        precisions = torch.stack(precisions)  # (C, D, D)
        logdets = torch.stack(logdets)  # (C,)
        log_priors = torch.stack(log_priors)  # (C,)

        diff = z_test[:, None, :] - means[None, :, :]  # (N, C, D)
        mahalanobis = torch.einsum("ncd,cde,nce->nc", diff, precisions, diff)
        scores = -0.5 * mahalanobis - 0.5 * logdets[None, :] + log_priors[None, :]
        y_pred = scores.argmax(dim=-1)
        accs.append((y_pred == y_test).float().mean().item())

    return accs
