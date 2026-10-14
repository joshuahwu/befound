import contextlib
import torch
import numpy as np
from befound.train.losses import get_batch_loss, get_batch_loss_per_dataset
from befound.train.metrics import (
    lda_rand_cv_torch,
    qda_rand_cv_torch,
    linear_regression_rand_cv_torch,
)
import torch.optim as optim
import tqdm
import time
import wandb
# from line_profiler import profile
from pathlib import Path
from befound.data.train_utils import prepare_batch
import copy
from befound.data.constants import OFFSETS_3D, DEFAULT_AUGMENTATIONS


class CyclicalBetaAnnealing(torch.nn.Module):
    def __init__(self, beta_max=1, len_cycle=100, R=0.5):
        self.beta_max = beta_max
        self.len_cycle = len_cycle
        self.R = R
        self.len_increasing = int(len_cycle * R)

    def get(self, epoch):
        remainder = (epoch - 1) % self.len_cycle
        if remainder >= self.len_increasing:
            beta = self.beta_max
        else:
            beta = self.beta_max * remainder / self.len_increasing

        return beta


def get_beta_schedule(schedule, beta):
    if schedule == "cyclical":
        print("Initializing cyclical beta annealing")
        beta_scheduler = CyclicalBetaAnnealing(beta_max=beta)
    else:
        print("No beta annealing selected")
        beta_scheduler = None

    return beta_scheduler


def get_optimizer_and_lr_scheduler(
    model, train_config, load_path=None, start_epoch=None
):
    """
    Loads in optimizer and learning rate schedulers
    """
    if train_config["optimizer"] == "adam":
        print("Initializing Adam optimizer ...")
        optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])
    elif train_config["optimizer"] == "adamw":
        print("Initializing AdamW optimizer ...")
        optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"])
    elif train_config["optimizer"] == "sgd":
        print("Initializing SGD optimizer ...")
        optimizer = optim.SGD(
            model.parameters(), lr=train_config["lr"], momentum=0.2, nesterov=True
        )
    else:
        raise ValueError("No valid optimizer selected")

    if train_config["lr_schedule"] == "cawr":
        print("Initializing cosine annealing w/warm restarts learning rate scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    elif train_config["lr_schedule"] is None:
        print("No learning rate scheduler selected")
        scheduler = None

    if load_path is not None:
        if Path("{}/checkpoints/epoch_{}.pth".format(load_path, start_epoch)).exists():
            checkpoint = torch.load(
                "{}/checkpoints/epoch_{}.pth".format(load_path, start_epoch)
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler = checkpoint["lr_scheduler"]

    return optimizer, scheduler


def predict_batch(model, data):
    data_i = {
        k: v for k, v in data.items() if k in ["x3d", "x2d", "dataset_id", "pe_indices"] + model.scrub_features
    }

    return model(data_i)


@contextlib.contextmanager
def _seeded_rng(seed):
    """Temporarily reseed torch's global RNG (CPU + all CUDA devices),
    restoring the prior state on exit -- torch.manual_seed sets global,
    process-wide state, so without saving/restoring it, a seeded test_epoch/
    test_epoch_per_dataset call (e.g. for a reproducible batch subset)
    would leak into everything that runs after it returns -- including the
    next train_epoch's own shuffling/augmentation/dropout randomness. A
    no-op (yields immediately) if seed is None.
    """
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# @profile
def train_epoch(config, model, loader, optimizer, scheduler, device="cuda", epoch=0, n_batches=None):
    model.train()

    kinematic_tree = loader.dataset.kinematic_tree
    default_offsets = loader.dataset.get_default_offsets(device)
    use_amp = config["train"].get("amp", False)

    with torch.enable_grad():
        epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}
        for batch_idx, data in enumerate(loader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            data = prepare_batch(
                data=data,
                dataset=loader.dataset,
                augment_dict=config["train"]["augmentations"],
                model_window=config["model"]["window"],
                device=device,
                get_2d=config["model"]["is_2d"],
                scrub_features=model.scrub_features,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                data_o = predict_batch(model, data)
                for k, v in data_o.items():
                    vals = v.values() if isinstance(v, dict) else [v]
                    for t in vals:
                        if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                            print(
                                f"non-finite in data_o[{k}] at epoch {epoch}, batch {batch_idx}"
                            )

                batch_loss = get_batch_loss(
                    model=model,
                    data=data,
                    data_o=data_o,
                    loss_scale=config["loss"],
                    # disentangle_config=config["disentangle"],
                    kinematic_tree=kinematic_tree,
                    default_offsets=default_offsets,
                    body_scale_used=loader.dataset.body_scale is not None,
                )

            # Update model parameters
            for param in model.parameters():
                param.grad = None

            batch_loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e4)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(loader))

            # Update the streaming scrubbers' running covariances/stats
            # (MovingAvgLeastSquares/MovingAverageFilter/
            # QuadraticDiscriminantFilter) -- forward() only reads these,
            # it never fits them, so without this call their weights
            # stay at their zero/identity initialization forever.
            if bool(model.disentangle):
                for method in model.disentangle.keys():
                    if method in ["moving_avg_lsq", "moving_avg", "qda"]:
                        for k in model.disentangle[method].keys():
                            model.disentangle[method][k].update(
                                data_o["mu"].detach().clone(),
                                data[k].detach().clone(),
                            )

            # Save loss and other metrics for the epoch
            epoch_metrics = {
                k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()
            }


        len_epoch = n_batches if n_batches is not None else len(loader)

        for k, v in epoch_metrics.items():
            epoch_metrics[k] = v.item() / len_epoch
            print(
                "====> Epoch: {} Average {} loss: {:.4f}".format(
                    epoch, k, epoch_metrics[k]
                )
            )

    return epoch_metrics


def _isolate_augmentation(key):
    """augment_dict with only `key` active (sourced from
    befound.data.constants.DEFAULT_AUGMENTATIONS, not the run's own config)
    and every other augmentation disabled. If `key`'s default config is a
    dict with a "prob" field, that field is forced to 1 so it always fires;
    boolean-style augmentations (currently just kpt_shuffle) are forced to
    True. `key`'s other parameters (e.g. kpt_ablation's max_ablate,
    rotation's pole_concentration) are left at their DEFAULT_AUGMENTATIONS
    value."""
    isolated = {k: None for k in DEFAULT_AUGMENTATIONS}
    value = copy.deepcopy(DEFAULT_AUGMENTATIONS[key])
    if isinstance(value, dict):
        # Dicts without "prob" (fps, rotation) always fire and keep their
        # parameters as-is
        if "prob" in value:
            value["prob"] = 1
    else:
        value = True
    isolated[key] = value
    return isolated


# @profile
def test_epoch_per_dataset(config, model, loader, device="cuda", epoch=0, n_batches=None, seed=None, get_z=False):
    """
    No-augmentation test pass; returns {dataset_name: {"jpe": ..., "root": ...}}
    plus a "dataset_id_decodability" entry (see below) and a "mu" entry --
    the concatenated (N, latent_dim) no-augmentation latents themselves, so
    callers can compare them against another pass's latents (e.g. the
    "2d_project" augmentation's own latents, in the sweep in train()).

    Unlike test_epoch (which combines every dataset's loss into one scalar
    per batch, then averages across batches), this accumulates each
    dataset's squared-error sums and sample count across the whole loader
    before dividing -- RodentDiskDataset lays out windows as contiguous
    per-dataset blocks, so an unshuffled loader's batches aren't an even
    mix of datasets and a per-batch average would be unreliable. NOTE this
    means that if `loader` itself is unshuffled (shuffle=False, no sampler)
    and `n_batches` is also given, the first n_batches will only ever cover
    whichever dataset(s) happen to come first in that fixed, contiguous
    layout -- not a representative cross-dataset subset. For a meaningful
    subset, `loader` needs shuffle=True (or a sampler) so each batch mixes
    datasets; `n_batches`/`seed` here only control how many such batches to
    consume and whether repeated calls see the same ones.

    Also accumulates every batch's mu (latent mean) and dataset_id across
    the whole loader, then runs the same style of cross-validated-classifier
    decodability check scrubvae.train.trainer uses for "ids" (see
    scrubvae.eval.metrics.lda_rand_cv/qda_rand_cv) on dataset_id instead --
    how accurately a held-out classifier can recover which dataset a
    window came from, given only its latent mu. High accuracy means
    dataset identity is still easily readable from the latent space (i.e.
    not yet scrubbed/disentangled away). Uses
    befound.train.metrics.lda_rand_cv_torch/qda_rand_cv_torch, a GPU-native
    reimplementation of the same two checks -- the originals are
    sklearn/numpy-based and too slow at this project's loader sizes.

    Parameters
    ----------
    n_batches : int, optional
        If given, stop after this many batches instead of consuming the
        whole loader -- see the note above about `loader` needing
        shuffle=True for this to be a representative subset.
    seed : int, optional
        If given, reseeds torch's global RNG for the duration of this call
        only (see _seeded_rng) -- repeated calls (e.g. once per augmentation
        in the sweep in train()) draw the identical subset/order of
        batches, without leaking that seed into anything that runs after
        this function returns.
    """
    model.eval()

    kinematic_tree = loader.dataset.kinematic_tree
    default_offsets = loader.dataset.get_default_offsets(device)
    dataset_names = loader.dataset.dataset_names
    use_amp = config["train"].get("amp", False)

    z = []
    dataset_id_list = []

    with torch.no_grad():#_seeded_rng(seed),
        epoch_metrics = {
            name: {"jpe": 0.0, "root": 0.0, "n": 0} for name in dataset_names
        }
        for batch_idx, data in enumerate(loader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            data = prepare_batch(
                data=data,
                dataset=loader.dataset,
                augment_dict={},
                model_window=config["model"]["window"],
                device=device,
                get_2d=config["model"]["is_2d"],
                scrub_features=model.scrub_features,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                data_o = predict_batch(model, data)

                z.append(data_o["mu"].detach())

                if "dataset_id" in data:
                    dataset_id_list.append(data["dataset_id"].detach())

                batch_loss = get_batch_loss_per_dataset(
                    data=data,
                    data_o=data_o,
                    dataset_names=dataset_names,
                    kinematic_tree=kinematic_tree,
                    default_offsets=default_offsets,
                )
            for name, vals in batch_loss.items():
                for metric, value in vals.items():
                    epoch_metrics[name][metric] += float(value)


        z = torch.cat(z, dim=0)

        epoch_metrics = {}
        for name, vals in epoch_metrics.items():
            epoch_metrics[name] = {
                "jpe": vals["jpe"] / vals["n"] if vals["n"] else float("nan"),
                "root": vals["root"] / vals["n"] if vals["n"] else float("nan"),
            }
            print(
                "====> Epoch: {} Dataset: {} Average jpe loss: {:.4f}, root loss: {:.4f}".format(
                    epoch, name, epoch_metrics[name]["jpe"], epoch_metrics[name]["root"]
                )
            )

        if dataset_id_list and len(dataset_names) > 1:
            dataset_id = torch.cat(dataset_id_list, dim=0)
            acc_lda = lda_rand_cv_torch(z, dataset_id, model.window, 5)
            acc_qda = qda_rand_cv_torch(z, dataset_id, model.window, 5)
            epoch_metrics["dataset_id_decodability"] = {
                "lda_acc_mean": np.mean(acc_lda),
                # "lda_acc_std": np.std(acc_lda),
                "qda_acc_mean": np.mean(acc_qda),
                # "qda_acc_std": np.std(acc_qda),
            }
            print(
                "====> Epoch: {} dataset_id decodability from mu -- "
                "lda acc: {:.4f}, qda acc: {:.4f}".format(
                    epoch,
                    epoch_metrics["dataset_id_decodability"]["lda_acc_mean"],
                    epoch_metrics["dataset_id_decodability"]["qda_acc_mean"],
                )
            )

    return epoch_metrics, z


# @profile
def test_epoch(
    config,
    model,
    loader,
    device="cuda",
    epoch=0,
    n_batches=None,
    seed=None,
    decode_key=None,
    get_z=False,
):
    """
    Parameters
    ----------
    n_batches : int, optional
        If given, stop after this many batches instead of consuming the
        whole loader -- for a large validation set, running every batch on
        every test pass (e.g. once per augmentation in the sweep in
        train()) is wasteful; a fixed-size subset is enough to compare
        parameter sets against each other.
    seed : int, optional
        If given, reseeds torch's global RNG for the duration of this call
        only (see _seeded_rng) -- repeated calls (e.g. once per augmentation
        in that same sweep) draw the identical subset/order of batches
        (controls both `loader`'s own shuffling -- RandomSampler/
        BalancedDatasetSampler, if shuffle=True -- and, with num_workers>0,
        the base seed DataLoader derives for its worker processes), without
        leaking that seed into anything that runs after this function
        returns.
    decode_key : str, optional
        If given (e.g. "fps" or "rotation" during the augmentation sweep in
        train()), accumulates every batch's latents and data[decode_key] across
        the loader and adds "{decode_key}_r2_mean" to
        the returned dict -- a fast cross-validated linear-regression check
        (see befound.train.metrics.linear_regression_rand_cv_torch) of how
        much of that augmentation's own sampled value is linearly readable
        from the latents. Only meaningful when that augmentation is actually active
        in config["train"]["augmentations"], since otherwise
        data[decode_key] doesn't vary.
    get_z : bool, optional
        If True, accumulates every batch's z (latent variables) across the loader and adds it
        to the returned dict as "z" (concatenated (N, latent_dim)) -- e.g.
        for comparing one augmentation's latents against another pass's
        (see the "2d_project" case in the sweep in train()).
    """
    model.eval()

    kinematic_tree = loader.dataset.kinematic_tree
    default_offsets = loader.dataset.get_default_offsets(device)
    use_amp = config["train"].get("amp", False)

    if decode_key is not None or get_z:
        z_list = []
    if decode_key is not None:
        target_list = []

    with torch.no_grad():# _seeded_rng(seed), 
        epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}
        n_batches_run = 0
        for batch_idx, data in enumerate(loader):
            if n_batches is not None and batch_idx >= n_batches:
                break
            n_batches_run += 1
            data = prepare_batch(
                data=data,
                dataset=loader.dataset,
                augment_dict=config["train"]["augmentations"],
                model_window=config["model"]["window"],
                device=device,
                get_2d=config["model"]["is_2d"],
                scrub_features=model.scrub_features,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                data_o = predict_batch(model, data)
                for k, v in data_o.items():
                    vals = v.values() if isinstance(v, dict) else [v]
                    for t in vals:
                        if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                            print(
                                f"non-finite in data_o[{k}] at epoch {epoch}, batch {batch_idx}"
                            )

                batch_loss = get_batch_loss(
                    model=model,
                    data=data,
                    data_o=data_o,
                    loss_scale=config["loss"],
                    # disentangle_config=config["disentangle"],
                    kinematic_tree=kinematic_tree,
                    default_offsets=default_offsets,
                    body_scale_used=loader.dataset.body_scale is not None,
                )

            if decode_key is not None or get_z:
                z_list.append(data_o["mu"].detach())
            if decode_key is not None:
                target_list.append(data[decode_key].detach())

            # Save loss and other metrics for the epoch
            epoch_metrics = {
                k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()
            }

        for k, v in epoch_metrics.items():
            epoch_metrics[k] = v.item() / n_batches_run
            print(
                "====> Epoch: {} Average {} loss: {:.4f}".format(
                    epoch, k, epoch_metrics[k]
                )
            )

        if (decode_key is not None or get_z) and z_list:
            z = torch.cat(z_list, dim=0)

            if decode_key is not None:
                target = torch.cat(target_list, dim=0)
                r2 = linear_regression_rand_cv_torch(
                    z, target, config["model"]["window"], 5
                )
                epoch_metrics[f"{decode_key}_r2_mean"] = np.mean(r2)
                # epoch_metrics[f"{decode_key}_r2_std"] = np.std(r2)
                print(
                    "====> Epoch: {} {} decodability from z -- R^2: {:.4f}".format(
                        epoch, decode_key, epoch_metrics[f"{decode_key}_r2_mean"]
                    )
                )

    if get_z:
        return epoch_metrics, z
    else:
        return epoch_metrics, None


# @profile
def train(config, model, loader_dict, run=None):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)
    # torch.backends.cudnn.benchmark = True
    # config = balance_disentangle(config, loader_dict["train"].dataset)

    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model,
        config["train"],
        config["model"]["load_model"],
        config["model"]["start_epoch"],
    )

    # Apply beta annealing if selected
    if "prior" in config["loss"].keys():
        beta_scheduler = get_beta_schedule(
            config["loss"]["prior"],
            config["train"]["beta_anneal"],
        )
    else:
        beta_scheduler = None

    # Epoch loop
    for epoch in tqdm.trange(
        config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
    ):
        # Update beta annealing if applicable
        if beta_scheduler is not None:
            config["loss"]["prior"] = beta_scheduler.get(epoch)
            print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

        starttime = time.time()
        # Train for an epoch
        train_metrics = train_epoch(
            config=config,
            model=model,
            loader=loader_dict["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
            epoch=epoch,
            n_batches=1000,
        )
        metrics = {"{}_train".format(k): v for k, v in train_metrics.items()}
        metrics["time"] = time.time() - starttime

        if (epoch % 5 == 0) and (epoch > 1):
            print("Saving model to folder: {}".format(config["out_path"]))
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
            )
            torch.save(
                {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler},
                "{}/checkpoints/epoch_{}.pth".format(config["out_path"], epoch),
            )

            # No augmentation: jpe/root per dataset
            per_dataset_metrics, z_base = test_epoch_per_dataset(
                config=config,
                model=model,
                loader=loader_dict["val"],
                device="cuda",
                epoch=epoch,
                n_batches=250,
                seed=epoch,
            )
            # Pulled out before the flatten below -- a raw (N, latent_dim)
            # tensor, not a {metric: value} dict like every other entry.
            # mu_no_aug = per_dataset_metrics.pop("mu")
            for name, vals in per_dataset_metrics.items():
                metrics.update({f"{name}_{k}_test": v for k, v in vals.items()})

            # One-by-one augmentation sweep: each isolated, prob/flag forced
            # on, aggregated across datasets (see _isolate_augmentation)
            config_test = copy.deepcopy(config)
            config_test["loss"] = {"jpe": 1, "root": 1}
            for key in DEFAULT_AUGMENTATIONS:
                # test_rotation/test_2d_project (per-config, default True for
                # configs that don't set them) let a config skip one of
                # these specific sweep entries entirely -- e.g. cheaper eval
                # for configs where that augmentation isn't meaningful.
                if key == "rotation" and not config["train"].get("test_rotation", True):
                    continue
                if key == "2d_project" and not config["train"].get("test_2d_project", True):
                    continue
                config_test["train"]["augmentations"] = _isolate_augmentation(key)
                # fps/rotation have their own sampled ground-truth value per
                # window (data["fps"]/data["rotation"]) -- check how much of
                # it leaks linearly into mu while that augmentation is active.
                decode_key = key if key in ("fps", "rotation") else None
                # 2d_project doesn't have its own continuous target to
                # decode -- instead, check whether its latents are
                # distinguishable from the no-augmentation latents above
                # (mu_no_aug) at all, via lda/qda group classification.
                test_metrics, test_z = test_epoch(
                    config=config_test,
                    model=model,
                    loader=loader_dict["val"],
                    device="cuda",
                    epoch=epoch,
                    n_batches=250,
                    seed=epoch,
                    decode_key=decode_key,
                    get_z = key == "2d_project",
                )

                if key == "2d_project":
                    mu_2d_project = test_z
                    z = torch.cat([z_base, mu_2d_project], dim=0)
                    group = torch.cat(
                        [
                            torch.zeros(len(z_base), dtype=torch.long, device=z.device),
                            torch.ones(len(mu_2d_project), dtype=torch.long, device=z.device),
                        ]
                    )
                    acc_lda = lda_rand_cv_torch(z, group, config["model"]["window"], 5)
                    acc_qda = qda_rand_cv_torch(z, group, config["model"]["window"], 5)
                    test_metrics["latent_lda_acc_mean"] = np.mean(acc_lda)
                    # test_metrics["latent_lda_acc_std"] = np.std(acc_lda)
                    test_metrics["latent_qda_acc_mean"] = np.mean(acc_qda)
                    # test_metrics["latent_qda_acc_std"] = np.std(acc_qda)
                    print(
                        "====> Epoch: {} 2d_project vs no-augmentation latents -- "
                        "lda acc: {:.4f}, qda acc: {:.4f}".format(
                            epoch,
                            test_metrics["latent_lda_acc_mean"],
                            test_metrics["latent_qda_acc_mean"],
                        )
                    )

                metrics.update({f"{key}_{k}_test": v for k, v in test_metrics.items()})

        wandb.log(metrics, epoch)

    return model
