from typing import Optional, Tuple
import torch
import torch.nn as nn

from module.model_component.modules import WN
from utils.model import sequence_mask


class PosteriorEncoder(nn.Module):
    """
    事後エンコーダー(Posterior Encoder)

    変分推論のための事後分布 q(z|x) をモデル化するエンコーダー。
    入力特徴量から潜在変数の平均と対数分散を推定し、
    再パラメータ化トリック(reparameterization trick)を用いてサンプリングを行う。
    VITSモデルにおいて、音響特徴量から潜在表現を抽出する役割を担う。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        """
        PosteriorEncoderの初期化

        Args:
            in_channels (int): 入力特徴量のチャンネル数（例: メルスペクトログラムの次元数）
            out_channels (int): 出力潜在変数のチャンネル数
            hidden_channels (int): 隠れ層のチャンネル数
            kernel_size (int): 畳み込みカーネルのサイズ
            dilation_rate (int): WaveNetブロックにおける膨張率(dilation rate)
            n_layers (int): WaveNetブロックの層数
            gin_channels (int, optional): グローバル条件付け用のチャンネル数。
                0の場合は条件付けなし。デフォルトは0。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        # 入力特徴量を隠れ層の次元に変換する1x1畳み込み
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # WaveNetスタイルの畳み込みエンコーダー（膨張畳み込みを使用）
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )

        # 平均と対数分散を出力するための射影層(out_channels * 2次元に変換)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播処理

        入力特徴量から潜在変数をサンプリングし、その統計量を返す。
        変分推論における事後分布 q(z|x) からのサンプリングを行う。

        Args:
            x (torch.Tensor): 入力特徴量 [バッチサイズ, in_channels, 時間長]
            x_lengths (torch.Tensor): 各サンプルの有効な時間長 [バッチサイズ]
            g (Optional[torch.Tensor]): グローバル条件付け用のテンソル
                                         [バッチサイズ, gin_channels, 1]
                                         Noneの場合は条件付けなし

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - z: サンプリングされた潜在変数 [バッチサイズ, out_channels, 時間長]
                - m: 潜在変数の平均 [バッチサイズ, out_channels, 時間長]
                - logs: 潜在変数の対数標準偏差 [バッチサイズ, out_channels, 時間長]
                - x_mask: パディング部分をマスクするマスク [バッチサイズ, 1, 時間長]
        """
        # 有効な時間長に基づいてマスクを生成(パディング部分を除外)
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        # 入力特徴量を隠れ層の次元に変換し、マスクを適用
        x = self.pre(x) * x_mask

        # WaveNetエンコーダーで特徴抽出(グローバル条件付けを適用可能)
        x = self.enc(x, x_mask, g=g)

        # 平均と対数標準偏差を計算(stats = [mean, log_std])
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # 再パラメータ化トリックを用いて潜在変数をサンプリング: z = μ + σ * ε (ε ~ N(0,1))
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask
