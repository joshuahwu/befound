import torch.nn as nn
import torch
from typing import Dict, Union, List


def find_latent_dim(
    window_size: int, kernel: int, num_layers: int, dilation=torch.ones(4)
):
    # Convolution math
    stride = 1 if any(dilation > 1) else 2
    layer_out = (
        lambda l_in, dil: (l_in + 2 * (kernel // 2) - dil * (kernel - 1) - 1) / stride
        + 1
    )

    l_out = window_size
    for i in range(num_layers):
        l_out = layer_out(l_out, dilation[i])

    return int(l_out)


def find_out_dim(latent_dim: int, kernel: int, num_layers: int, dilation=torch.ones(4)):
    # Convolution math
    stride = 1 if any(dilation > 1) else 2
    layer_out = (
        lambda l_in, dil: (l_in - 1) * stride
        - 2 * (kernel // 2)
        + dil * (kernel - 1)
        + 1
    )
    l_out = latent_dim
    for i in range(num_layers):
        l_out = layer_out(l_out, dilation[-i])

    return int(l_out)


class ResConv1DBlock(nn.Module):
    def __init__(
        self,
        n_in,
        n_state,
        n_out,
        dilation=1,
        activation="prelu",
        norm="LN",
        transpose=False,
    ):
        super().__init__()
        padding = dilation
        self.norm = norm
        self.transpose = transpose
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=n_in, eps=1e-6, affine=True
            )
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "prelu":
            self.activation1 = nn.PReLU()
            self.activation2 = nn.PReLU()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        if transpose:
            self.conv1 = nn.ConvTranspose1d(
                in_channels=n_in,
                out_channels=n_state,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
            self.conv2 = nn.ConvTranspose1d(
                in_channels=n_state,
                out_channels=n_out,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.conv1 = nn.Conv1d(
                in_channels=n_in,
                out_channels=n_state,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
            self.conv2 = nn.Conv1d(
                in_channels=n_state,
                out_channels=n_out,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        depth,
        dilation_growth_rate=1,
        reverse_dilation=False,
        activation="prelu",
        norm=None,
        transpose=False,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in,
                n_in,
                n_in,
                dilation=dilation_growth_rate**d,
                activation=activation,
                norm=norm,
                transpose=transpose,
            )
            for d in range(depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Conv1DEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        depth: int = 3,
        stride: int = 2,
        dilation: int = 3,
        activation: str = "prelu",
        normalization: str = "LN",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.ds = nn.Conv1d(
            in_channels,
            hidden_dim,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2 if stride % 2 == 0 else stride // 2 + 1,
        )

        self.res_block = Resnet1D(
            n_in=hidden_dim,
            depth=depth,
            dilation_growth_rate=dilation,
            activation=activation,
            norm=normalization,
            transpose=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ds(x)
        x = self.res_block(x)
        return x


class Conv1DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: List[int],
        strides: Union[list, int] = 2,
        depth: int = 2,
        dilation: int = 1,
        activation: str = "relu",
        normalization: str = "LN",
        window_size: int = 51,
        out_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = hidden_dim[-1]
        self.n_ds = len(hidden_dim)
        self.window_size = window_size
        self.latent_T = window_size // (2**self.n_ds)

        if isinstance(strides, int):
            strides = [strides] * self.n_ds
        else:
            assert len(strides) == self.n_ds

        layers = []
        for i in range(self.n_ds):
            layers.append(
                Conv1DEncoderBlock(
                    in_channels=in_channels if i == 0 else hidden_dim[i - 1],
                    hidden_dim=hidden_dim[i],
                    depth=depth,
                    stride=strides[i],
                    dilation=dilation,
                    activation=activation,
                    normalization=normalization,
                )
            )
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Conv1d(
            hidden_dim[-1],
            self.out_channels,
            kernel_size=out_kernel_size,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor, pe_indices=None) -> torch.Tensor:
        # bs, n_points, D_in, T = x.shape
        # x = x.reshape(bs, n_points * D_in, T)
        x = self.backbone(x)
        x = self.out(x)
        return x


class Conv1DDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        stride: int = 2,
        depth: int = 3,
        dilation: int = 3,
        activation: str = "relu",
        normalization: str = "LN",
        out_kernel_size: int = 7,
    ):
        super().__init__()

        self.res_block = Resnet1D(
            n_in=in_channels,
            depth=depth,
            dilation_growth_rate=dilation,
            activation=activation,
            norm=normalization,
            transpose=True,
        )

        self.upsamp = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="linear", align_corners=False),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=out_kernel_size,
                stride=1,
                padding=out_kernel_size // 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x)
        x = self.upsamp(x)
        return x


class Conv1DDecoder(nn.Module):
    def __init__(
        self,
        out_channels: Union[int, dict],
        hidden_dim: List[int],
        latent_T: int,
        strides: Union[list, int] = 2,
        depth: int = 2,
        dilation: int = 1,
        activation: str = "relu",
        normalization: str = "LN",
        in_kernel_size: int = 3,
        out_kernel_size: int = 5,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = hidden_dim[0]
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.latent_T = latent_T
        self.n_us = len(hidden_dim)

        if isinstance(strides, int):
            strides = [strides] * self.n_us
        else:
            assert len(strides) == self.n_us

        self.input = nn.Conv1d(
            self.in_channels,
            hidden_dim[0],
            kernel_size=in_kernel_size,
            stride=1,
            padding=1,
        )

        if isinstance(self.out_channels, dict):
            self.head_in_channels = hidden_dim[-1]
        elif isinstance(self.out_channels, int):
            self.head_in_channels = self.out_channels

        layers = []
        for i in range(self.n_us):
            layers.append(
                Conv1DDecoderBlock(
                    in_channels=hidden_dim[i],
                    hidden_dim=(
                        self.out_channels if (i == self.n_us - 1) else hidden_dim[i + 1]
                    ),
                    depth=depth,
                    stride=strides[i],
                    dilation=dilation,
                    activation=activation,
                    normalization=normalization,
                    out_kernel_size=4,
                )
            )
        self.backbone = nn.Sequential(*layers)

        if isinstance(self.out_channels, dict):
            self.out = nn.ModuleDict({k: nn.Conv1d(
                self.hidden_dim[-1],
                self.out_channels[k],
                kernel_size=out_kernel_size,
                stride=1,
                padding=0,
            ) for k in self.out_channels.keys()})

        elif isinstance(self.out_channles, int):
            self.out = nn.Conv1d(
                self.out_channels,
                self.out_channels,
                kernel_size=out_kernel_size,
                stride=1,
                padding=0,
            )
        else:
            raise ValueError("requires number of output channels to be set")
        

    def forward(self, x: torch.Tensor, dataset_id: torch.Tensor = None, pe_indices=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for Conv1DDecoder with optional dataset-specific heads.

        Parameters
        ----------
        x : torch.Tensor
            Input latent features of shape (bs, in_channels, T)
        dataset_id : torch.Tensor, optional
            Dataset indices for each sample (bs,) when using dict heads.
            Required if self.out_channels is a dict (multi-dataset mode).
        pe_indices : optional
            Positional encoding indices (unused)

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            If out_channels is int: single output tensor (bs, out_channels, T')
            If out_channels is dict: dict[dataset_name] -> output tensor for that dataset
        """
        bs, _, T = x.shape
        x = self.input(x)
        x = self.backbone(x)

        # Single output head (backward compatible)
        if isinstance(self.out_channels, int):
            x = torch.tanh(self.out(x))
            return x

        # Multiple dataset-specific heads - keep outputs separated by dataset
        if isinstance(self.out_channels, dict):
            if dataset_id is None:
                raise ValueError("dataset_id required when using dict output heads")

            # Initialize output dict for all datasets
            output = {}

            # Route each sample to its dataset's head
            dataset_names = list(self.out_channels.keys())
            for dataset_idx, dataset_name in enumerate(dataset_names):
                # Find samples belonging to this dataset
                mask = dataset_id == dataset_idx
                if not mask.any():
                    output[dataset_name] = torch.empty(0, device=x.device)
                    continue

                # Extract samples for this dataset
                x_dataset = x[mask]  # (n_samples, channels, T)

                # Apply dataset-specific head
                out_dataset = torch.tanh(self.out[dataset_name](x_dataset))
                output[dataset_name] = out_dataset

            return output

        raise ValueError("out_channels must be int or dict")
