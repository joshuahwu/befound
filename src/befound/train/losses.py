import torch.nn.functional as F
import torch
from befound.data import quaternion as qtn
from befound.data import fwd_kin_cont6d_torch
import numpy as np
import wandb
from befound.model.vae import ResVAE2D, CIResVAE2D

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


def mpjpe_loss(pose, x_hat, kinematic_tree, offsets, root_hat=None):
    """
    Mean per joint position error
    """
    # if root == None:
    #     root = torch.zeros((x.shape[0], 3), device=x.device)
    if root_hat == None:
        root_hat = torch.zeros_like(pose[..., 0, :])
    # pose = fwd_kin_cont6d_torch(
    #     x, kinematic_tree, offsets, root_pos=root, do_root_R=True, eps=1e-8
    # )
    # pose = x
    pose_hat = fwd_kin_cont6d_torch(
        x_hat.reshape((-1,) + x_hat.shape[-2:]),
        kinematic_tree,
        offsets.reshape((-1,) + offsets.shape[-2:]),
        root_pos=root_hat.reshape((-1, 3)),
        do_root_R=True,
        eps=1e-8,
    ).reshape(pose.shape)

    loss = torch.sum((pose - pose_hat) ** 2)
    loss = loss / (pose.shape[0] * pose.shape[-1] * pose.shape[-2])
    return loss


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


def get_batch_loss(
    model, data, data_o, loss_scale, kinematic_tree=None, offsets_sum=None
):
    is_2d = isinstance(model, ResVAE2D) or isinstance(model, CIResVAE2D)
    if is_2d:
        window = data["x2d"].shape[-3]
        batch_size = data["x2d"].shape[0]
    else:
        window = data["x3d"].shape[-3]
        batch_size = data["x3d"].shape[0]
    batch_loss = {}

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
                    (data_o["x2d"] - data_o["x2d"][:, :, 0:1]) * offsets_sum,
                    data["target_pose"][:, :, :, :2],
                )
                / batch_size
                / data_o["x2d"].shape[-2]
                / data_o["x2d"].shape[-1]
            )
        else:
            batch_loss["jpe"] = mpjpe_loss(
                data[
                    "target_pose"
                ],  # data["x6d"].reshape(-1, *data["x6d"].shape[-2:]),
                data_o["x6d"],
                kinematic_tree,
                data["offsets"],
            )

    if "root" in loss_scale.keys():
        if is_2d:
            batch_loss["root"] = torch.nn.MSELoss(reduction="sum")(
                data_o["x2d"][:, :, 0] * offsets_sum, data["root"][:, :, :2]
            )
        else:
            batch_loss["root"] = (
                torch.nn.MSELoss(reduction="sum")(
                    data_o["root"] * offsets_sum, data["root"]
                )
                / batch_size
            )

    ## TODO: In testing for hierarchical vae
    # if "orthogonal_cov" in loss_scale.keys():
    #     batch_loss["orthogonal_cov"] = hierarchical_orthogonal_loss(*data_o["L"])

    batch_loss["total"] = sum(
        [loss_scale[k] * batch_loss[k] for k in batch_loss.keys() if loss_scale[k] != 0]
    )

    return batch_loss
