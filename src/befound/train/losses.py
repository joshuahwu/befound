import torch.nn.functional as F
import torch
from befound.data import quaternion as qtn
from befound.data import fwd_kin_cont6d_torch
from befound.data.constants import BODY_SCALE_LOSS_FACTOR
import numpy as np
import wandb

LN2PI = np.log(2 * np.pi)


def balance_disentangle(config, dataset):
    # Balance disentanglement losses
    if config["disentangle"]["balance_loss"]:
        print("Balancing disentanglement losses")
        for k in config["disentangle"]["features"]:
            var = torch.sqrt((dataset[:][k].std(dim=0) ** 2).sum()).detach().numpy()
            config["loss"][k] /= var
            if k + "_gr" in config["loss"].keys():
                config["loss"][k + "_gr"] /= var

        print("Finished disentanglement loss balancing...")
        print(config["loss"])
    return config


def _gaussian_log_density_unsummed(z, mu, logvar):
    """First step of Gaussian log-density computation, without summing over dimensions.

    Assumes a diagonal noise covariance matrix.

    Code taken from:
    Whiteway, Matthew R., et al. "Partitioning variability in animal behavioral
    videos using semi-supervised variational autoencoders." PLoS computational
    biology 17.9 (2021): e1009439.
    """
    diff_sq = (z - mu) ** 2
    inv_var = torch.exp(-logvar)
    return -0.5 * (inv_var * diff_sq + logvar + LN2PI)


def total_correlation(z, mu, L):
    """Estimate total correlation in a batch.

    Compute the expectation over a batch of:

    E_j [log(q(z(x_j))) - log(prod_l q(z(x_j)_l))]

    We ignore the constant as it does not matter for the minimization. The constant should be
    equal to (n_dims - 1) * log(n_frames * dataset_size).

    Code modified from https://github.com/julian-carpenter/beta-TCVAE/blob/master/nn/losses.py

    Code taken from:
    Whiteway, Matthew R., et al. "Partitioning variability in animal behavioral
    videos using semi-supervised variational autoencoders." PLoS computational
    biology 17.9 (2021): e1009439.

    Parameters
    ----------
    z : :obj:`torch.Tensor`
        sample of shape (n_frames, n_dims)
    mu : :obj:`torch.Tensor`
        mean parameter of shape (n_frames, n_dims)
    logvar : :obj:`torch.Tensor`
        log variance parameter of shape (n_frames, n_dims)

    Returns
    -------
    :obj:`torch.Tensor`
        total correlation for batch, scalar value

    """
    logvar = torch.log(
        torch.matmul(L, torch.transpose(L, dim0=-2, dim1=-1)).diagonal(dim1=-1, dim2=-2)
    )
    # Compute log(q(z(x_j)|x_i)) for every sample/dimension in the batch, which is a tensor of
    # shape (n_frames, n_dims). In the following comments,
    # (n_frames, n_frames, n_dims) are indexed by [j, i, l].
    # z[:, None]: (n_frames, 1, n_dims)
    # mu[None, :]: (1, n_frames, n_dims)
    # logvar[None, :]: (1, n_frames, n_dims)
    log_qz_prob = _gaussian_log_density_unsummed(
        z[:, None].detach(), mu[None, :], logvar[None, :]
    )

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(x_j)_l|x_i))) + constant) for each
    # sample in the batch, which is a vector of size (batch_size,).
    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1, keepdim=False),  # logsumexp over batch
        dim=1,  # sum over gaussian dims
        keepdim=False,
    )

    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2, keepdim=False),  # sum over gaussian dims
        dim=1,  # logsumexp over batch
        keepdim=False,
    )
    return torch.mean(log_qz - log_qz_product)


def rotation_loss(x, x_hat, eps=1e-7):
    """
    Geodesic rotation loss of two 6D rotation representations
    """
    assert x.shape[-1] == 6
    assert x_hat.shape[-1] == 6
    batch_size = x.shape[0]
    m1 = qtn.cont6d_to_matrix(x).view((-1, 3, 3))
    m2 = qtn.cont6d_to_matrix(x_hat).view((-1, 3, 3))

    m = torch.bmm(m1, m2.permute(0, 2, 1))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.clamp(cos, -1 + eps, 1 - eps)
    theta = torch.acos(cos).sum() / batch_size

    return theta


def stable_rotation_loss(x, x_hat, eps=1e-7):
    """
    Geodesic rotation loss of two 6D rotation representations

    More numerically stable
    """
    assert x.shape[-1] == 6
    assert x_hat.shape[-1] == 6
    m1 = qtn.cont6d_to_matrix(x).view((-1, 3, 3))
    m2 = qtn.cont6d_to_matrix(x_hat).view((-1, 3, 3))

    sin = torch.linalg.matrix_norm(m2 - m1) / (2**1.5)
    sin = torch.clamp(sin, -1 + eps, 1 - eps)
    return 2 * torch.asin(sin).sum()


def prior_loss(mu, logvar):
    KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KL_div / mu.shape[0]


def _mpjpe_sq_err(pose, x_hat, kinematic_tree, offsets, root_hat=None):
    """
    Squared position error between a ground-truth pose and the pose
    reconstructed from predicted continuous-6D rotations, already divided by
    (n_keypts * 3) -- a fixed per-dataset constant, not something that
    varies batch to batch, so there's no reason to defer it to callers.

    Returns (squared_error_sum, n_terms), where n_terms = pose.shape[0] (the
    batch count) is the only piece that actually varies across batches/
    datasets, so callers can accumulate both before dividing.
    """
    if root_hat is None:
        root_hat = torch.zeros_like(pose[..., 0, :])

    # offsets may be a fixed (N_K, 3) canonical skeleton (broadcast across the
    # batch inside fwd_kin_cont6d_torch) or a per-sample (..., N_K, 3) tensor
    # that needs collapsing to match the flattened (B*W, N_K, 6) x_hat.
    offsets_arg = (
        offsets if offsets.dim() == 2 else offsets.reshape((-1,) + offsets.shape[-2:])
    )

    pose_hat = fwd_kin_cont6d_torch(
        x_hat.reshape((-1,) + x_hat.shape[-2:]),
        kinematic_tree,
        offsets_arg,
        root_pos=root_hat.reshape((-1, 3)),
        do_root_R=True,
        eps=1e-8,
    ).reshape(pose.shape)

    sq_err_sum = torch.sum((pose - pose_hat) ** 2) / (pose.shape[-1] * pose.shape[-2])
    n_terms = pose.shape[0]
    return sq_err_sum, n_terms


def mpjpe_loss(
    pose,
    x_hat,
    kinematic_tree,
    offsets,
    dataset_id=None,
    dataset_names=None,
    root_hat=None,
):
    """
    Mean per joint position error.

    Supports two modes:

    Single dataset (original behavior): `x_hat` is a single tensor
    (B, W, N_K, 6), `kinematic_tree`/`offsets` describe that one dataset.

    Multi dataset: `x_hat` is a Dict[str, torch.Tensor] keyed by dataset name
    (as returned by a dict-headed decoder), each value shaped
    (B_ds, W, N_K_ds, 6) for just that dataset's samples. `pose` is the
    single, zero-padded (B, W, N_MAX, 3) target_pose tensor spanning all
    datasets in the batch; each dataset's samples are selected via
    `dataset_id`/`dataset_names` and sliced down to that dataset's real
    keypoint count before computing the error. `kinematic_tree`/`offsets`
    must then be dicts keyed by the same dataset names.

    Parameters
    ----------
    dataset_id : torch.Tensor (B,), required in multi-dataset mode
        Integer dataset index per sample, matching the enumeration of
        `dataset_names`.
    dataset_names : List[str], required in multi-dataset mode
    """
    if isinstance(x_hat, dict):
        if dataset_names is None:
            raise ValueError("dataset_names is required when x_hat is a dict")
        if dataset_id is None and len(dataset_names) != 1:
            raise ValueError(
                "dataset_id is required when x_hat is a dict with more than one dataset"
            )

        sq_err_sum = None
        n_terms = 0
        for ds_idx, dataset_name in enumerate(dataset_names):
            x_hat_ds = x_hat.get(dataset_name)
            if x_hat_ds is None or x_hat_ds.numel() == 0:
                continue

            n_keypts_ds = x_hat_ds.shape[-2]
            # With a single dataset, dataset_id may be omitted entirely --
            # every sample belongs to it, so skip masking.
            if dataset_id is None:
                pose_ds = pose[..., :n_keypts_ds, :]
            else:
                mask = dataset_id == ds_idx
                pose_ds = pose[mask][..., :n_keypts_ds, :]

            # offsets may be a dict of per-dataset canonical (N_K, 3) offsets,
            # or a single padded tensor spanning all datasets (like pose)
            if isinstance(offsets, dict):
                offsets_ds = offsets[dataset_name]
            elif dataset_id is None:
                offsets_ds = offsets[..., :n_keypts_ds, :]
            else:
                offsets_ds = offsets[mask][..., :n_keypts_ds, :]

            ds_sq_err_sum, ds_n = _mpjpe_sq_err(
                pose_ds, x_hat_ds, kinematic_tree[dataset_name], offsets_ds
            )
            sq_err_sum = (
                ds_sq_err_sum if sq_err_sum is None else sq_err_sum + ds_sq_err_sum
            )
            n_terms += ds_n

        if sq_err_sum is None:
            raise ValueError("No dataset in x_hat had any samples in this batch")

        return sq_err_sum / n_terms

    sq_err_sum, n_terms = _mpjpe_sq_err(pose, x_hat, kinematic_tree, offsets, root_hat)
    return sq_err_sum / n_terms


# def mpjpe_loss(x, x_hat):
#     """
#     Mean per joint position error
#     """
#     # if root == None:
#     #     root = torch.zeros((x.shape[0], 3), device=x.device)
#     x_centered = x - x[..., 0:1, :]
#     x_hat_centered = x_hat - x_hat[..., 0:1, :]

#     loss = torch.sum((x_centered - x_hat_centered) ** 2)
#     loss = loss / (x.shape[0] * x.shape[-1] * x.shape[-2])
#     return loss


def time_consistency_loss(x, x_hat, kinematic_tree, offsets):
    root_hat = torch.zeros_like(offsets[..., 0, 0, :])
    pose = fwd_kin_cont6d_torch(
        x[:, 0],
        kinematic_tree,
        offsets[:, 0],
        root_pos=root_hat,
        do_root_R=False,
        eps=1e-8,
    )

    pose_hat = fwd_kin_cont6d_torch(
        x_hat[:, 0],
        kinematic_tree,
        offsets[:, 0],
        root_pos=root_hat,
        do_root_R=False,
        eps=1e-8,
    )

    loss = torch.sum((pose - pose_hat) ** 2)
    loss = loss / (pose.shape[0] * pose.shape[-1] * pose.shape[-2])

    return loss


def direct_lsq_loss(z, y, bias=False):
    if bias:
        z = torch.column_stack((z, torch.ones(z.shape[0], 1, device="cuda")))
    zz = z.T @ z
    zy = z.T @ y
    yhat = z @ torch.linalg.solve(zz, zy)
    return torch.nn.MSELoss(reduction="sum")(yhat, y)


def get_batch_loss_per_dataset(
    data,
    data_o,
    dataset_names,
    kinematic_tree,
    default_offsets=None,
):
    """
    Per-dataset jpe/root accumulators (squared-error sum + term count) for
    one batch, 3D-only (no is_2d support, unlike get_batch_loss).

    data["target_pose"]/data["root"] and data_o["x6d"]/data_o["root"] are
    all already body-scale-normalized (RodentDiskDataset.__getitem__
    normalizes pose at the source, and the model is trained end-to-end on
    that normalized pose), so no body_scale handling is needed here.

    Unlike get_batch_loss's "jpe"/"root" (which combine every dataset in the
    batch into one scalar), this keeps each dataset's contribution separate
    so a caller can accumulate sq_err/n across an entire loader -- needed
    because with an unshuffled loader, datasets aren't evenly mixed within
    a single batch (RodentDiskDataset lays out windows as contiguous
    per-h5_key blocks), so a per-batch average would be unreliable.

    Returns
    -------
    Dict[str, Dict[str, float/int]]
        {dataset_name: {"jpe", "root", "n"}} -- jpe and
        root are both already divided by their own per-sample term
        count (n_keypts*3 for jpe, nothing for root -- see
        get_batch_loss's "root"), so both share the same remaining
        denominator: n, this dataset's sample count (B_id) in this batch.
    """
    dataset_id = data.get("dataset_id", None)
    batch_loss = {}
    for ds_idx, dataset_name in enumerate(dataset_names):
        x_hat_ds = data_o["x6d"].get(dataset_name)
        if x_hat_ds is None or x_hat_ds.numel() == 0:
            batch_loss[dataset_name] = {"jpe": 0.0, "root": 0.0, "n": 0}
            continue

        n_keypts_ds = x_hat_ds.shape[-2]
        if dataset_id is None:
            mask = torch.ones(
                data["target_pose"].shape[0], dtype=torch.bool, device=x_hat_ds.device
            )
        else:
            mask = dataset_id == ds_idx

        pose_ds = data["target_pose"][mask][..., :n_keypts_ds, :]
        offsets_ds = (
            default_offsets[dataset_name]
            if default_offsets is not None
            else data["offsets"][mask][..., :n_keypts_ds, :]
        )
        jpe_sq_err, n = _mpjpe_sq_err(
            pose_ds, x_hat_ds, kinematic_tree[dataset_name], offsets_ds
        )

        root_sq_err = 0.0
        root_hat_ds = data_o["root"].get(dataset_name)
        if root_hat_ds is not None and root_hat_ds.numel() > 0:
            root_sq_err = torch.sum((root_hat_ds - data["root"][mask]) ** 2)

        batch_loss[dataset_name] = {
            "jpe": jpe_sq_err,
            "root": root_sq_err,
            "n": n,
        }

    return batch_loss


def get_batch_loss(
    model,
    data,
    data_o,
    loss_scale,
    kinematic_tree=None,
    default_offsets=None,
    body_scale_used=False,
):
    """
    data["target_pose"]/data["root"] and data_o["x6d"]/data_o["root"] are
    all already body-scale-normalized (RodentDiskDataset.__getitem__
    normalizes pose at the source, and the model is trained end-to-end on
    that normalized pose), so no further body_scale handling is needed to
    compute jpe/root correctly here.

    body_scale_used : bool
        True if the dataset that produced `data` was built with
        scale_body=True (pass loader.dataset.body_scale is not None). If
        so, "jpe" and "root" are each multiplied by
        befound.data.constants.BODY_SCALE_LOSS_FACTOR to restore roughly
        their pre-normalization magnitude -- normalization otherwise
        shrinks their relative weight against e.g. "prior" compared to
        previous, unnormalized runs.
    """
    is_2d = model.is_2d
    if is_2d:
        window = data["x2d"].shape[-3]
        batch_size = data["x2d"].shape[0]
    else:
        window = data["x3d"].shape[-3]
        batch_size = data["x3d"].shape[0]
    batch_loss = {}

    ## Scrubbing losses
    # methods_dict = disentangle_config["method"]
    # for method, disentangle_keys in model.disentangle.items():
    #     method = model.disentangle[disentangle_key]
    for method, scrubbers in model.disentangle.items():
        # import pdb; pdb.set_trace()
        num_keys = len(scrubbers.keys())
        for key in scrubbers.keys():
            # if "linear" in methods_dict.keys():
            #     latent = data_o["disentangle"]["linear"][key]["z_null"]
            # else:
            #     latent = data_o["mu"]

            if method == "moving_avg_lsq" and key + "_mals" in loss_scale.keys():
                batch_loss[key + "_mals"] = (
                    scrubbers[key].evaluate_loss(
                        data_o["disentangle"][method][key][0],
                        data_o["disentangle"][method][key][1],
                        data[key],
                    )
                    / batch_size
                )

            if method == "qda" and key + "_qda" in loss_scale.keys():
                batch_loss[key + "_qda"] = (
                    scrubbers[key].evaluate_loss(data_o["mu"], data[key]) / batch_size
                )

            if method == "direct_lsq" and key + "_lsq" in loss_scale.keys():
                batch_loss[key + "_lsq"] = direct_lsq_loss(
                    data_o["mu"], data[key], bias=loss_scale[key + "_lsq"] < 0
                )

            if method == "grad_reversal" and key + "_gr" in loss_scale.keys():
                batch_loss[key + "_gr"] = 0
                for gr_e in data_o["disentangle"][method][key]:
                    if key == "ids":
                        batch_loss[key + "_gr"] += torch.nn.CrossEntropyLoss(
                            reduction="sum"
                        )(gr_e, data[key].ravel().long())
                    else:
                        batch_loss[key + "_gr"] += torch.nn.MSELoss(reduction="sum")(
                            gr_e, data[key]
                        )

                    batch_loss[key + "_gr"] = (
                        batch_loss[key + "_gr"]
                        / len(data_o["disentangle"][method][key])
                        / num_keys
                        / batch_size
                    )

            if method == "moving_avg" and key + "_ma" in loss_scale.keys():
                batch_loss[key + "_ma"] = (
                    scrubbers[key].evaluate_loss(data_o["mu"], data[key]) / batch_size
                )

            if method == "adversarial_net" and key + "_an" in loss_scale.keys():
                var_ind = model.disentangle_keys.index(key)
                z_aug, var_aug = scrubbers[key].shuffle(
                    data_o["mu"], data_o["var"], var_ind
                )
                y_pred = scrubbers[key].forward(z_aug, var_aug)
                y = (
                    torch.tensor([0, 1], device=data_o["mu"].device)[:, None]
                    .repeat(1, batch_size)
                    .ravel()
                )
                y = torch.nn.functional.one_hot(y, 2).float()

                ce = torch.nn.CrossEntropyLoss(reduction="sum")
                batch_loss[key + "_an"] = 0
                for y_ens in y_pred:
                    batch_loss[key + "_an"] += ce(y_ens, y)

                batch_loss[key + "_an"] /= -(len(y_pred) * batch_size)

    if "prior" in loss_scale.keys():
        # if type(data_o["mu"]) is tuple:
        #     # For if you have multiple latent spaces (e.g. hierarchical)
        #     batch_loss["prior"] = 0
        #     for mu, L in zip(data_o["mu"], data_o["L"]):
        #         batch_loss["prior"] += prior_loss(mu, L)
        # else:
        if model.prior == "gaussian":
            batch_loss["prior"] = prior_loss(data_o["mu"], data_o["logvar"])
        elif model.prior == "beta":
            p = torch.distributions.Beta(
                torch.ones_like(data_o["alpha"]),
                torch.ones_like(data_o["beta"]),
            )
            batch_loss["prior"] = (
                torch.distributions.kl_divergence(data_o["beta_dist"], p).sum(-1).sum()
                / batch_size
            )

    if "jpe" in loss_scale.keys():
        if is_2d:
            batch_loss["jpe"] = (
                torch.nn.MSELoss(reduction="sum")(
                    (data_o["x2d"] - data_o["x2d"][:, :, 0:1]),
                    data["target_pose"][:, :, :, :2],
                )
                / batch_size
                / data_o["x2d"].shape[-2]
                / data_o["x2d"].shape[-1]
            )
        else:
            offsets_arg = default_offsets if default_offsets is not None else data["offsets"]

            batch_loss["jpe"] = mpjpe_loss(
                data["target_pose"],
                data_o["x6d"],
                kinematic_tree,
                offsets_arg,
                dataset_id=data.get("dataset_id", None),
                dataset_names=model.dataset_names,
            )
            if body_scale_used:
                batch_loss["jpe"] = batch_loss["jpe"] * BODY_SCALE_LOSS_FACTOR

    if "root" in loss_scale.keys():
        if is_2d:
            #TODO: update for multiple datasets?
            batch_loss["root"] = torch.nn.MSELoss(reduction="sum")(
                data_o["x2d"][:, :, 0], data["root"][:, :, :2]
            )
        else:
            # data_o["root"] is a dict of per-dataset (B_ds, W, 3) predictions
            # (already body-scale-normalized); scatter each into a single
            # (B, W, 3) tensor matching data["root"]'s sample order (via
            # dataset_id)
            root_hat = torch.zeros_like(data["root"])
            dataset_id = data.get("dataset_id", None)
            for ds_idx, dataset_name in enumerate(model.dataset_names):
                root_hat_ds = data_o["root"].get(dataset_name)
                if root_hat_ds is None or root_hat_ds.numel() == 0:
                    continue
                # With a single dataset, dataset_id may be omitted entirely --
                # every sample belongs to it, so skip masking.
                if dataset_id is None:
                    root_hat = root_hat_ds
                else:
                    mask = dataset_id == ds_idx
                    # root_hat_ds comes from the model's output -- under
                    # torch.autocast it's the active autocast dtype (e.g.
                    # bf16), while root_hat matches data["root"]'s plain
                    # float32 (the loss below is computed in fp32, the usual
                    # AMP pattern) -- masked assignment requires an exact
                    # dtype match, unlike ordinary arithmetic, which would
                    # auto-promote.
                    root_hat[mask] = root_hat_ds.to(root_hat.dtype)

            batch_loss["root"] = (
                torch.nn.MSELoss(reduction="sum")(root_hat, data["root"]) / batch_size
            )
        if body_scale_used:
            batch_loss["root"] = batch_loss["root"] * BODY_SCALE_LOSS_FACTOR

    batch_loss["total"] = sum(
        [loss_scale[k] * batch_loss[k] for k in batch_loss.keys() if loss_scale[k] != 0]
    )

    return batch_loss
