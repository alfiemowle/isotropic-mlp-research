import torch
import torch.nn as nn


class IsotropicTanhActivation(nn.Module):
    """
    f(x) = tanh(||x||) * x_hat
    with a linear approximation for ||x|| < epsilon.

    Input:  [B, C, H, W]
    Norm is taken over channels -> [B, 1, H, W]
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

        # ||x|| over channels: [B, 1, H, W]
        magnitude = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)

        # small-ball mask
        small = magnitude < self.epsilon

        # outside the small ball:
        # tanh(||x||) * x_hat = (tanh(||x||)/||x||) * x
        nonlinear_scale = torch.tanh(magnitude) / magnitude.clamp_min(self.epsilon)
        nonlinear_out = nonlinear_scale * x

        # inside the small ball, use linear approximation:
        # tanh(r) ~ r  => tanh(r)/r ~ 1
        linear_out = x

        return torch.where(small, linear_out, nonlinear_out)


class IsotropicLog1pActivation(nn.Module):
    """
    f(x) = log(1 + ||x||) * x_hat
    with a linear approximation for ||x|| < epsilon.

    Since log(1+r)/r -> 1 as r -> 0, the local linear map is also x.
    """

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

        magnitude = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
        small = magnitude < self.epsilon

        nonlinear_scale = torch.log1p(magnitude) / magnitude.clamp_min(self.epsilon)
        nonlinear_out = nonlinear_scale * x

        linear_out = x

        return torch.where(small, linear_out, nonlinear_out)


class IsotropicConvolution(nn.Module):
    """
    Conv2d followed by isotropic activation over channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation: str = "tanh",
        epsilon: float = 1e-3,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if activation == "tanh":
            self.activation = IsotropicTanhActivation(epsilon=epsilon)
        elif activation in {"log", "log1p"}:
            self.activation = IsotropicLog1pActivation(epsilon=epsilon)
        else:
            raise ValueError("activation must be one of {'tanh', 'log1p'}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))