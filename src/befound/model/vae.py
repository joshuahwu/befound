import torch
import torch.nn as nn
import torch.nn.functional as F
from befound.model.attention import SetAttentiveBlock
from befound.model.conv import Conv1DEncoder, Conv1DDecoder
from typing import Dict, Union, List


def _one_hot_or_zero(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    One-hot encode idx (any shape), except rows outside [0, num_classes) get
    an all-zero row instead of raising -- for a discrete scrub_feature like
    dataset_id, samples from an eval-only dataset (appended past the
    training class range in get.data_and_model) fall here, since they were
    deliberately excluded from this disentangling class range.
    """
    idx = idx.ravel().long()
    valid = (idx >= 0) & (idx < num_classes)
    out = torch.zeros(idx.shape[0], num_classes, dtype=torch.long, device=idx.device)
    out[valid] = F.one_hot(idx[valid], num_classes)
    return out


class ChannelInvariantEncoder(nn.Module):
    def __init__(
        self,
        channel_encoder: Conv1DEncoder,
        num_heads: int = 2,
        query_size: int | List[int] = 8,
        n_keypts: int | None = None,
        dataset_names: List[str] = [],
        latent_dim: int = 64,
        use_be: bool = False,
    ):
        super().__init__()
        self.channel_encoder = channel_encoder
        self.data_dim = channel_encoder.in_channels
        self.chan_out_dim = channel_encoder.out_channels
        self.n_ds = channel_encoder.n_ds
        self.window_size = self.channel_encoder.window_size
        self.latent_T = self.channel_encoder.latent_T
        self.n_keypts = n_keypts
        self.dataset_names = dataset_names

        if isinstance(query_size, int):
            query_size = [query_size] * self.n_ds
        self.query_size = query_size  # Num_inds M for each block (size of learnable query array [1, M, D])

        attention_blocks = []
        for i in range(self.n_ds):
            block = self.channel_encoder.backbone[i]
            chan_out_dim = block.ds.out_channels
            attention_blocks.append(
                SetAttentiveBlock(
                    dim_in=chan_out_dim,
                    dim_out=chan_out_dim,
                    num_heads=num_heads,
                    num_inds=query_size[i],
                )
            )
        self.attention_blocks = nn.Sequential(*attention_blocks)

        self.latent_dim = latent_dim
        self.n_channels_out = query_size[-1] * self.chan_out_dim

        self.use_be = use_be
        if use_be:
            self.be = nn.Embedding(7, channel_encoder.hidden_dim[0])

    def _forward(self, x: torch.Tensor, pe_indices=None, **kwargs) -> torch.Tensor:
        """Process sets of data points in a channel-invariant manner

        Args:
            x (torch.Tensor): [B, n_points, D, T], D is the data dimension (e.g., 3 for 3D Cartesian coordinates)

        Returns:
            out: torch.Tensor: [B, n_points, D_out, T']
            hidden_feats: List[torch.Tensor], [B*T', query_size, D]
        """

        bs, n_points, D_in, T = x.shape
        x = x.reshape(bs * n_points, D_in, T)

        query_feats = []
        # import pdb; pdb.set_trace()
        for i in range(self.n_ds):
            x = self.channel_encoder.backbone[i](x)  # --> [B*N, D, T//ds]

            x = x.reshape(bs, n_points, *x.shape[1:])  # --> [B, N, D, T//ds=T']
            x = x.permute(0, 3, 1, 2).flatten(0, 1)  # --> [B*T', N, D]

            x, h, attn1, attn2 = self.attention_blocks[i](x)

            query_feats.append(
                h.reshape(bs, -1, *h.shape[1:]).permute(0, 2, 3, 1).flatten(0, 1)
            )  # --> [B*M, D, T']
            x = (
                x.reshape(bs, -1, *x.shape[1:]).permute(0, 2, 3, 1).flatten(0, 1)
            )  # [B*T', D, N] --> [B*N, D, T']

            if i == 0 and self.use_be and pe_indices is not None:
                pe = self.be(pe_indices.to(x.device)).unsqueeze(-1)
                x = x.reshape(bs, -1, *x.shape[1:])
                x += pe
                x = x.flatten(0, 1)

        out = query_feats[-1] #x if self.chan_mix_last else 
        out = self.channel_encoder.out(out)
        out = out.reshape(bs, -1, *out.shape[1:])  # [B, N or M, D_out, T']
        out = out.flatten(1, 2)  # [B, (N or M)*D_out, T']
        return out

    def forward(self, x: torch.Tensor, pe_indices=None, dataset_id=None, **kwargs):
        """
        Forward pass with optional per-dataset processing.

        Parameters
        ----------
        x : torch.Tensor
            Input [B, n_points, D, T]
        pe_indices : optional
            Positional encoding indices
        dataset_id : torch.Tensor, optional
            Dataset IDs for each sample for per-dataset processing
        """
        # Single dataset (backward compatible)
        if dataset_id is None:
            # If padded, index to actual keypoints
            max_actual = max(self.n_keypts.values())
            x = x[:, :max_actual, :, :]

            out = self._forward(x, pe_indices, **kwargs)
            return out

        # Multi-dataset: process each dataset separately and concatenate
        # outputs = []
        dataset_indices = sorted(set(dataset_id.tolist()))
        # x itself stays float32 here (it's an input tensor, not an op
        # output autocast would have touched), but self._forward's conv/
        # attention ops run through autocast and return its dtype -- so
        # out's dtype has to come from the active autocast dtype, not x's,
        # or the masked write below mismatches under torch.autocast.
        out_dtype = (
            torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else x.dtype
        )
        out = torch.zeros(
            (x.shape[0], self.n_channels_out, self.latent_T), device=x.device, dtype=out_dtype
        )
        for ds_idx in dataset_indices:
            mask = dataset_id == ds_idx

            if not mask.any():
                continue

            n_keypts_ds = self.n_keypts[self.dataset_names[ds_idx]]
            x_dataset = x[mask, :n_keypts_ds]

            # Extract pe_indices for this dataset
            pe_idx_dataset = None
            if pe_indices is not None:
                pe_idx_dataset = pe_indices[mask, :n_keypts_ds]

            # Forward pass
            out[mask] = self._forward(x_dataset, pe_idx_dataset, **kwargs)
            # import pdb; pdb.set_trace()
        #     outputs.append(out_dataset)

        # out = torch.cat(outputs, dim=0)
        return out

    def __str__(self):
        return "CI_MIEncoder"


class VAE(nn.Module):
    def __init__(self, prior="gaussian"):
        super(VAE, self).__init__()
        self.prior = prior
        if prior == "gaussian":
            self.dist_params = ["mu", "logvar"]
        elif prior == "beta":
            self.dist_params = ["alpha", "beta"]
        elif prior is None:
            self.dist_params = ["mu"]
        return self

    def device(self):
        return next(self.parameters()).device

    def sampling(self, mu, logvar):
        """Reparameterization trick

        Parameters
        ----------
        mu : torch.tensor
            Batch_size x latent dimensions
        L : torch.tensor
            Batch_size x latent dimensions x latent dimensions. Lower triangular or diagonal matrix.
        """
        eps = torch.randn_like(mu)
        return torch.exp(0.5 * logvar).mul(eps).add_(mu)

    def forward(self, data):
        data_o = self.encode(data)

        # Reparameterize
        if self.prior == "gaussian":
            z = (
                self.sampling(data_o["mu"], data_o["logvar"])
                if self.training
                else data_o["mu"]
            )
        elif self.prior == "beta":
            beta_dist = torch.distributions.Beta(data_o["alpha"], data_o["beta"])
            data_o["beta_dist"] = beta_dist
            z = beta_dist.rsample() * 2 - 1
        elif self.prior is None:
            z = data_o["mu"]

        data_o["z"] = z

        # Pass entire data dict to decode for access to dataset_id
        data_o.update(self.decode(z, data))

        if hasattr(self, "disentangle"):
            # Running disentangle
            data_o["disentangle"] = {}
            if "linear" in self.disentangle.keys():
                data_o["disentangle"]["linear"] = {
                    k: model(data_o["mu"])
                    for k, model in self.disentangle["linear"].items()
                }

            # Forward pass through all scrubbers if necessary
            for method, module_dict in self.disentangle.items():
                if method == "linear":
                    ## Placeholder for if we reimplement in the future
                    continue
                else:
                    data_o["disentangle"][method] = {}
                    for k, model in module_dict.items():
                        if "linear" in self.disentangle.keys():
                            latent = data_o["disentangle"]["linear"][k]["z_null"]
                        else:
                            latent = data_o["mu"]
                        if method == "adversarial_net":
                            data_o["disentangle"][method][k] = model(latent, data_o["var"])
                        else:
                            data_o["disentangle"][method][k] = model(latent)

        return data_o


class ResVAEBase(VAE):
    """Base class for ResVAE variants (2D and 3D) with common encode/decode logic.

    Subclasses should set:
    - self.encoder: The encoder module (ChannelInvariantEncoder or Conv1DEncoder)
    - self.decoder: The Conv1DDecoder module
    - self.is_2d: Whether to process x2d (True) or x3d (False)
    - self.fc_mu, self.fc_logvar, self.fc_decoder: Bottleneck layers
    """

    def _init_disentangle(
        self,
        scrubber_dict=None,
        conditional_dim=0,
        discrete_classes=None,
    ):
        """Shared disentangle/conditional setup for all ResVAE variants.

        Mirrors scrubvae.model.residual.ResVAE.__init__: builds self.disentangle
        as an nn.ModuleDict of {method: {feature: scrubber_module}} so VAE.forward
        can run the scrubbers, and records the conditional bookkeeping decode()
        needs to concatenate disentangled/conditional variables onto z.
        """
        self.conditional_dim = conditional_dim
        self.discrete_classes = discrete_classes

        if scrubber_dict is not None:
            self.scrub_features = [feat for k in scrubber_dict.keys() for feat in scrubber_dict[k].keys()]
            self.disentangle = nn.ModuleDict()
            for k, v in scrubber_dict.items():
                self.disentangle[k] = nn.ModuleDict(v)
        else:
            self.scrub_features = []
            self.disentangle = nn.ModuleDict(nn.ModuleDict())

        self.mi_estimator = None

    def encode(self, data):
        """Common encode logic for all ResVAE variants (2D and 3D)."""
        data_o = {}

        # Select data key based on is_2d
        x_key = "x2d" if self.is_2d else "x3d"
        x = data[x_key].permute(0, 2, 3, 1)  # [B, D, T, n_keypts] -> [B, n_keypts, D, T]

        # Pass dataset_id for per-dataset processing
        dataset_id = data.get("dataset_id", None)
        pe_indices = data.get("pe_indices", None)

        out = self.encoder(x, pe_indices=pe_indices, dataset_id=dataset_id)
        out = out.flatten(1, 2)
        data_o["mu"] = self.fc_mu(out)
        data_o["logvar"] = self.fc_logvar(out)
        return data_o

    def decode(self, z, data):
        """Common decode logic for all ResVAE variants (2D and 3D)."""
        data_o = {}

        if self.conditional_dim > 0:
            # Concatenating disentanglement/conditional variables together
            data_o["var"] = [
                (
                    _one_hot_or_zero(data[k], len(self.discrete_classes[k]))
                    if k in self.discrete_classes.keys()
                    else data[k]
                )
                for k in self.scrub_features
            ]
            data_o["var"] = torch.cat(data_o["var"], dim=-1)
            z = torch.cat([z, data_o["var"]], dim=-1)

        out = self.fc_decoder(z)
        out = out.reshape(z.shape[0], self.hidden_dim[-1], -1)

        # Pass dataset_id to decoder if using multi-dataset heads
        dataset_id = data.get("dataset_id", None)
        decoder_out = self.decoder(out, dataset_id=dataset_id)

        if self.is_2d:
            # 2D output: x2d only
            if isinstance(decoder_out, dict):
                data_o["x2d"] = {}
                for dataset_name, out_dataset in decoder_out.items():
                    data_o["x2d"][dataset_name] = out_dataset.permute(0, 2, 1).reshape(
                        out_dataset.shape[0], -1, self.n_keypts, 2
                    )
            else:
                data_o["x2d"] = decoder_out.permute(0, 2, 1).reshape(
                    z.shape[0], -1, self.n_keypts, 2
                )
        else:
            # 3D output: x6d and root
            if isinstance(decoder_out, dict):
                data_o["x6d"] = {}
                data_o["root"] = {}
                for dataset_name, out_dataset in decoder_out.items():
                    out_perm = out_dataset.permute(0, 2, 1)
                    data_o["root"][dataset_name] = out_perm[..., -3:]
                    x6d = F.normalize(
                        out_perm[..., :-3].reshape(out_dataset.shape[0], -1, self.n_keypts[dataset_name], 2, 3), dim=-1
                    )
                    data_o["x6d"][dataset_name] = x6d.reshape(x6d.shape[:-2] + (6,))
            else:
                out = decoder_out.permute(0, 2, 1)
                data_o["root"] = out[..., -3:]
                x6d = F.normalize(
                    out[..., :-3].reshape(z.shape[0], -1, self.n_keypts, 2, 3), dim=-1
                )
                data_o["x6d"] = x6d.reshape(x6d.shape[:-2] + (6,))

        return data_o


class CIResVAE(ResVAEBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: List[int] = [128, 256, 512],
        latent_dim: int = 64,
        depth: int = 2,
        query_size: int = 8,
        activation: str = "prelu",
        out_kernel_size: int = 7,
        n_keypts: Dict[str, int] = {},
        dataset_names: List[str] = [],
        prior: str = "gaussian",
        window_size: int = 51,
        is_2d: bool = False,
        use_be: bool = False,
        scrubber_dict=None,
        conditional_dim: int = 0,
        discrete_classes=None,
    ):
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.z_dim = latent_dim
        self.out_kernel_size = out_kernel_size
        self.query_size = query_size
        self.depth = depth
        self.n_keypts = n_keypts
        self.out_channels = out_channels
        self.window_size = window_size
        self.window = window_size
        self.is_2d = is_2d
        self.dataset_names = dataset_names
        self.use_be = use_be
        self._init_disentangle(
            scrubber_dict=scrubber_dict,
            conditional_dim=conditional_dim,
            discrete_classes=discrete_classes,
        )
        channel_encoder = Conv1DEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            strides=[2] * len(hidden_dim),
            depth=depth,
            dilation=1,
            activation=activation,
            normalization="LN",
            window_size=window_size,
            out_kernel_size=3,
        )

        self.encoder = ChannelInvariantEncoder(
            channel_encoder=channel_encoder,
            num_heads=4,
            query_size=query_size,
            n_keypts=n_keypts,
            dataset_names=dataset_names,
            latent_dim=latent_dim,
            use_be=use_be,
        )
        self.latent_T = self.encoder.latent_T
        fc_in = self.encoder.n_channels_out * self.latent_T

        self.fc_mu = nn.Linear(fc_in, latent_dim)  # M*D_out --> D_latent
        self.fc_logvar = nn.Linear(fc_in, latent_dim)

        self.fc_decoder = nn.Linear(
            latent_dim + self.conditional_dim, self.encoder.latent_T * self.hidden_dim[-1]
        )  # D_latent (+ scrub_features) --> M*D_out
        self.decoder = Conv1DDecoder(
            out_channels=self.out_channels,
            hidden_dim=hidden_dim[::-1],
            latent_T=self.latent_T,
            strides=[2] * len(hidden_dim),
            depth=depth,
            dilation=1,
            activation=activation,
            normalization="LN",
            out_kernel_size=out_kernel_size,
        )

class ResVAE(ResVAEBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: List[int] = [128, 256, 512],
        latent_dim: int = 64,
        depth: int = 2,
        activation: str = "prelu",
        out_kernel_size: int = 7,
        n_keypts: Dict[str, int] = {"wu_iclr25":18},
        dataset_names: List[str] = [],
        prior: str = "gaussian",
        window_size: int = 51,
        is_2d: bool = False,
        scrubber_dict=None,
        conditional_dim: int = 0,
        discrete_classes=None,
    ):
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.z_dim = latent_dim
        self.out_kernel_size = out_kernel_size
        self.out_channels = out_channels
        self.depth = depth
        self.window_size = window_size
        self.window = window_size
        self.n_keypts = n_keypts
        self.is_2d = is_2d
        self.dataset_names = dataset_names
        self._init_disentangle(
            scrubber_dict=scrubber_dict,
            conditional_dim=conditional_dim,
            discrete_classes=discrete_classes,
        )

        self.encoder = Conv1DEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            strides=[2] * len(hidden_dim),
            depth=depth,
            dilation=1,
            activation=activation,
            normalization="LN",
            window_size=window_size,
            out_kernel_size=3,
        )
        self.latent_T = self.encoder.latent_T
        fc_in = hidden_dim[-1] * self.latent_T

        self.fc_mu = nn.Linear(fc_in, latent_dim)  # M*D_out --> D_latent
        self.fc_logvar = nn.Linear(fc_in, latent_dim)

        self.fc_decoder = nn.Linear(
            latent_dim + self.conditional_dim, self.encoder.latent_T * self.hidden_dim[-1]
        )  # D_latent (+ scrub_features) --> M*D_out
        self.decoder = Conv1DDecoder(
            out_channels=self.out_channels,
            hidden_dim=hidden_dim[::-1],
            latent_T=self.latent_T,
            strides=[2] * len(hidden_dim),
            depth=depth,
            dilation=1,
            activation=activation,
            normalization="LN",
            out_kernel_size=out_kernel_size,
        )