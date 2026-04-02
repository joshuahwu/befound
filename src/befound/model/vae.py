import torch
import torch.nn as nn
import torch.nn.functional as F
from befound.model.attention import SetAttentiveBlock
from befound.model.conv import Conv1DEncoder, Conv1DDecoder
from typing import Dict, Union, List


class ChannelInvariantEncoder(nn.Module):
    def __init__(
        self,
        channel_encoder: Conv1DEncoder,
        num_heads: int = 2,
        query_size: int | List[int] = 8,
        chan_mix_last: bool = False,
        num_points: int | None = None,
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
        self.chan_mix_last = chan_mix_last and num_points is not None
        if self.chan_mix_last:
            self.n_channels_out = num_points * self.chan_out_dim
        else:
            self.n_channels_out = query_size[-1] * self.chan_out_dim

        self.use_be = use_be
        if use_be:
            self.be = nn.Embedding(6, channel_encoder.hidden_dim[0])

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

        out = x if self.chan_mix_last else query_feats[-1]
        out = self.channel_encoder.out(out)
        out = out.reshape(bs, -1, *out.shape[1:])  # [B, N or M, D_out, T']
        out = out.flatten(1, 2)  # [B, (N or M)*D_out, T']
        return out

    def forward(self, x: torch.Tensor, pe_indices=None, **kwargs):
        out = self._forward(x, pe_indices, **kwargs)  # --> [B, (N or M)*D_out, T']

        # mu = self.fc_mu(out.flatten(1, 2))
        # sigma = self.fc_sigma(out.flatten(1, 2))
        # out = out.permute(0, 2, 1)

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

        data_o.update(self.decode(z))

        return data_o


class CIResVAE(VAE):
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
        n_keypts: int = 18,
        prior: str = "gaussian",
        window_size: int = 51,
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
            chan_mix_last=False,
            num_points=None,
            latent_dim=latent_dim,
            use_be=True,
        )
        self.latent_T = self.encoder.latent_T

        if self.encoder.chan_mix_last:
            fc_in = self.encoder.n_channels_out * self.latent_T
        else:
            fc_in = self.encoder.n_channels_out * self.latent_T

        self.fc_mu = nn.Linear(fc_in, latent_dim)  # M*D_out --> D_latent

        self.fc_logvar = nn.Linear(fc_in, latent_dim)

        self.fc_decoder = nn.Linear(
            latent_dim, self.encoder.latent_T * self.hidden_dim[-1]
        )  # D_latent --> M*D_out
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

    def encode(self, data):
        data_o = {}
        out = self.encoder(data["x3d"].permute(0, 2, 3, 1))
        out = out.flatten(1, 2)
        data_o["mu"] = self.fc_mu(out)
        data_o["logvar"] = self.fc_logvar(out)
        return data_o

    def decode(self, z):
        data_o = {}
        out = self.fc_decoder(z)
        out = out.reshape(z.shape[0], self.hidden_dim[-1], -1)

        out = self.decoder(out).permute(0, 2, 1)
        data_o["root"] = out[..., -3:]

        x6d = F.normalize(
            out[..., :-3].reshape(z.shape[0], -1, self.n_keypts, 2, 3), dim=-1
        )

        data_o["x6d"] = x6d.reshape(x6d.shape[:-2] + (6,))

        return data_o


class ResVAE(VAE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: List[int] = [128, 256, 512],
        latent_dim: int = 64,
        depth: int = 2,
        activation: str = "prelu",
        out_kernel_size: int = 7,
        n_keypts: int = 18,
        prior: str = "gaussian",
        window_size: int = 51,
    ):
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.z_dim = latent_dim
        self.out_kernel_size = out_kernel_size
        self.out_channels = out_channels
        self.depth = depth
        self.window_size = window_size
        self.n_keypts = n_keypts

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
            latent_dim, self.encoder.latent_T * self.hidden_dim[-1]
        )  # D_latent --> M*D_out
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

    def normalize_root(self, root):
        norm_root = root - self.arena_size[0]
        norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        return norm_root

    def inv_normalize_root(self, norm_root):
        root = 0.5 * (norm_root + 1) * (self.arena_size[1] - self.arena_size[0])
        root += self.arena_size[0]
        return root

    def encode(self, data):
        data_o = {}
        out = self.encoder(data["x3d"].permute(0, 2, 3, 1).flatten(1, 2))
        out = out.flatten(1, 2)
        data_o["mu"] = self.fc_mu(out)
        data_o["logvar"] = self.fc_logvar(out)
        return data_o

    def decode(self, z):
        data_o = {}
        out = self.fc_decoder(z)
        out = out.reshape(z.shape[0], self.hidden_dim[-1], -1)
        out = self.decoder(out).permute(0, 2, 1)
        data_o["root"] = out[..., -3:]

        x6d = F.normalize(
            out[..., :-3].reshape(z.shape[0], -1, self.n_keypts, 2, 3), dim=-1
        )

        data_o["x6d"] = x6d.reshape(x6d.shape[:-2] + (6,))

        return data_o

class ResVAE2D(VAE):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: List[int] = [128, 256, 512],
        latent_dim: int = 64,
        depth: int = 2,
        activation: str = "prelu",
        out_kernel_size: int = 7,
        n_keypts: int = 18,
        prior: str = "gaussian",
        window_size: int = 51,
    ):
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.z_dim = latent_dim
        self.out_kernel_size = out_kernel_size
        self.out_channels = out_channels
        self.depth = depth
        self.window_size = window_size
        self.n_keypts = n_keypts

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
            latent_dim, self.encoder.latent_T * self.hidden_dim[-1]
        )  # D_latent --> M*D_out
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

    def encode(self, data):
        data_o = {}
        out = self.encoder(data["x2d"].permute(0, 2, 3, 1).flatten(1, 2))
        out = out.flatten(1, 2)
        data_o["mu"] = self.fc_mu(out)
        data_o["logvar"] = self.fc_logvar(out)
        return data_o

    def decode(self, z):
        data_o = {}
        out = self.fc_decoder(z)
        out = out.reshape(z.shape[0], self.hidden_dim[-1], -1)
        data_o["x2d"] = self.decoder(out).permute(0, 2, 1).reshape(z.shape[0], self.window_size, self.n_keypts, -1)
        return data_o


class CIResVAE2D(VAE):
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
        n_keypts: int = 18,
        prior: str = "gaussian",
        window_size: int = 51,
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
            chan_mix_last=False,
            num_points=None,
            latent_dim=latent_dim,
            use_be=True,
        )
        self.latent_T = self.encoder.latent_T

        if self.encoder.chan_mix_last:
            fc_in = self.encoder.n_channels_out * self.latent_T
        else:
            fc_in = self.encoder.n_channels_out * self.latent_T

        self.fc_mu = nn.Linear(fc_in, latent_dim)  # M*D_out --> D_latent

        self.fc_logvar = nn.Linear(fc_in, latent_dim)

        self.fc_decoder = nn.Linear(
            latent_dim, self.encoder.latent_T * self.hidden_dim[-1]
        )  # D_latent --> M*D_out
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

    def encode(self, data):
        data_o = {}
        out = self.encoder(data["x2d"].permute(0, 2, 3, 1))
        out = out.flatten(1, 2)
        data_o["mu"] = self.fc_mu(out)
        data_o["logvar"] = self.fc_logvar(out)
        return data_o

    def decode(self, z):
        data_o = {}
        out = self.fc_decoder(z)
        out = out.reshape(z.shape[0], self.hidden_dim[-1], -1)

        data_o["x2d"] = self.decoder(out).permute(0, 2, 1).reshape(z.shape[0], self.window_size, self.n_keypts, -1)

        return data_o