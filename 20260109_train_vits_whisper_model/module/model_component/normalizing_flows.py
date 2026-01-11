from typing import Optional
import torch
import torch.nn as nn

from module.model_component.modules import (
    Flip,
    ResidualCouplingLayer,
    TransformerCouplingLayer,
    WN,
)
from module.model_component.transformer import FFT


class ResidualCouplingBlock(nn.Module):
    """
    Residual Coupling Block (複数のResidual Coupling Layerを積層したブロック)

    正規化フロー(Normalizing Flow)において、複数のアフィン結合層と
    Flip層を交互に配置したブロックです。

    構造:
        [ResidualCouplingLayer -> Flip] × n_flows

    Flip層はチャネルの順序を反転させることで、各Coupling Layerで
    異なるチャネルが変換されるようにし、表現力を向上させます。

    パラメータ共有:
        share_parameter=Trueの場合、全てのCoupling Layerで同じWaveNetエンコーダを
        共有し、パラメータ数を削減します。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        """
        Residual Coupling Blockの初期化

        Args:
            channels (int): 入力チャネル数(2で割り切れる必要がある)
            hidden_channels (int): WaveNetエンコーダの隠れ層チャネル数
            kernel_size (int): WaveNetの畳み込みカーネルサイズ
            dilation_rate (int): WaveNetの膨張率(dilation rate)
            n_layers (int): WaveNetの層数
            n_flows (int, optional): Coupling Layerの積層数。デフォルトは4。
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                0の場合は単一話者、0以外の場合は多話者モデル。
                デフォルトは0。
            share_parameter (bool, optional): 全てのCoupling Layerで
                WaveNetパラメータを共有するか。Trueの場合、パラメータ数を
                大幅に削減できます。デフォルトはFalse。
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        # パラメータ共有用のWaveNetエンコーダを作成
        # share_parameter=Trueの場合、全てのCoupling Layerでこれを共有
        self.wn = (
            WN(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels,
                p_dropout=0.0,
            )
            if share_parameter
            else None
        )

        # n_flows個のCoupling LayerとFlip層を交互に配置
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,  # 対数スケールを省略し、平均のみ使用
                    wn_sharing_parameter=self.wn,  # パラメータ共有
                )
            )
            # Flip層: チャネルの順序を反転
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Residual Coupling Blockの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
            x_mask (torch.Tensor): パディングマスク (B, 1, T)
            g (Optional[torch.Tensor], optional): 話者埋め込みベクトル
                (B, gin_channels)。デフォルトはNone。
            reverse (bool, optional): 逆変換を行うかどうか。
                デフォルトはFalse。

        Returns:
            torch.Tensor: 変換後のテンソル (B, channels, T)
        """
        if not reverse:
            # 順方向: 各フローを順々に適用
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # 逆方向: 各フローを逆順で適用
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class TransformerCouplingBlock(nn.Module):
    """
    Transformer Coupling Block (複数のTransformer Coupling Layerを積層したブロック)

    ResidualCouplingBlockと同様の構造ですが、WaveNetの代わりにFFTを
    使用しています。これにより、より長い文脈を捉えることができます。

    構造:
        [TransformerCouplingLayer -> Flip] x n_flows

    パラメータ共有:
        share_parameter=Trueの場合、全てのCoupling Layerで同じFFTエンコーダを
        共有し、パラメータ数を削減します。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        """
        Transformer Coupling Blockの初期化

        Args:
            channels (int): 入力チャネル数(2で割り切れる必要がある)
            hidden_channels (int): FFTエンコーダの隠れ層チャネル数
            filter_channels (int): FFT内のFFN層の中間チャネル数
            n_heads (int): FFTのマルチヘッドアテンションのヘッド数
            n_layers (int): FFTのTransformer層数
            kernel_size (int): FFT内のFFN層の畳み込みカーネルサイズ
            p_dropout (float): ドロップアウト率
            n_flows (int, optional): Coupling Layerの積層数。デフォルトは4。
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                0の場合は単一話者、0以外の場合は多話者モデル。
                デフォルトは0。
            share_parameter (bool, optional): 全てのCoupling Layerで
                FFTパラメータを共有するか。Trueの場合、パラメータ数を
                大幅に削減できます。デフォルトはFalse。
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        # パラメータ共有用のFFTエンコーダを作成
        # share_parameter=Trueの場合、全てのCoupling Layerでこれを共有
        self.wn = (
            FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,  # 話者条件付けを有効化
                gin_channels=gin_channels,
            )
            if share_parameter
            else None
        )

        # n_flows個のCoupling LayerとFlip層を交互に配置
        for i in range(n_flows):
            self.flows.append(
                TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,  # 対数スケールを省略し、平均のみ使用
                    wn_sharing_parameter=self.wn,  # パラメータ共有
                    gin_channels=self.gin_channels,
                )
            )
            # Flip層: チャネルの順序を反転
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Transformer Coupling Blockの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
            x_mask (torch.Tensor): パディングマスク (B, 1, T)
            g (Optional[torch.Tensor], optional): 話者埋め込みベクトル
                (B, gin_channels)。FFTのisflow=Trueで話者条件付けに使用。
                デフォルトはNone。
            reverse (bool, optional): 逆変換を行うかどうか。
                デフォルトはFalse。

        Returns:
            torch.Tensor: 変換後のテンソル (B, channels, T)
        """
        if not reverse:
            # 順方向: 各フローを順々に適用
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # 逆方向: 各フローを逆順で適用
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
