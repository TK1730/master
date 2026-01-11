import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """チャネル単位のLayer Normalization。

    入力テンソルのチャンネル軸(C)に沿って正規化を行うレイヤー。
    通常 (B, C, T) や (B, C, ...) の形状を想定し、出力は入力と同じ形状です。

    Args:
        channels (int): 正規化対象のチャンネル数 (C)。
        eps (float, optional): 分散の計算で使用する小さい値（数値安定化）。デフォルトは 1e-5。

    Input shape:
        (batch, channels, ...)

    Returns:
        torch.Tensor: 正規化後のテンソル（入力と同じ形状）。

    Example:
        ln = LayerNorm(80)
        out = ln(x)  # x: (B, 80, T)

    注意:
        実装では内部的にテンソルを転置してチャンネル軸を最後にしてから
        torch.nn.functional.layer_norm を使っています。
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """レイヤー正規化を適用

        Args:
            x (torch.Tensor): 入力テンソル (B, C, ...)

        Returns:
            torch.Tensor: 正規化後のテンソル（入力と同じ形状）。
        """
        x = F.layer_norm(
            x.mT,
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps
        )
        return x.mT
