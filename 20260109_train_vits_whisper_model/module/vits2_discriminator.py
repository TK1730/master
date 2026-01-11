from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

from module.model_component.modules import LRELU_SLOPE
from module.model_component.normalization import LayerNorm
from utils.model import get_padding


class DiscriminatorP(torch.nn.Module):
    """
    Period-based Discriminator (周期ベース識別器)

    1D音声波形を2D表現に変換し、特定の周期パターンを識別します。
    Multi-Period Discriminatorの一部として使用されます。

    アーキテクチャ:
        - 入力を指定された周期で2D表現に変換
        - 5層の2D畳み込み層 (1→32→128→512→1024→1024チャンネル)
        - 各層でLeaky ReLU活性化を適用
        - 最終的に真偽判定スコアを出力

    特徴:
        - 周期的なパターンに敏感
        - 各層の特徴マップを保存 (Feature Matching Loss用)
        - Weight NormまたはSpectral Normを使用可能
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        """
        Period-based Discriminatorを初期化します。

        Args:
            period (int): 2D変換時の周期
                音声を[batch, channels, time//period, period]に変換
            kernel_size (int): 畳み込みカーネルサイズ. Defaults to 5.
            stride (int): 畳み込みストライド. Defaults to 3.
            use_spectral_norm (bool): Spectral Normalizationを使用するか
                Falseの場合はWeight Normを使用. Defaults to False.
        """
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm

        # 正規化関数を選択 (Weight Norm or Spectral Norm)
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        # 畳み込み層のリスト
        # チャンネル数: 1→32→128→512→1024→1024
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        # 最終出力層: 1024→1チャンネル (真偽スコア)
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        順伝播処理

        1D音声を指定された周期で2D表現に変換し、真偽判定を行います。

        Args:
            x (torch.Tensor): 入力音声波形
                Shape: [batch, 1, time]

        Returns:
            tuple: 以下の要素を含むタプル
                - 判定スコア (torch.Tensor): 真偽判定結果
                    Shape: [batch, *] (flatten後)
                - 特徴マップリスト (list[torch.Tensor]): 各層の出力
                    Feature Matching Loss計算に使用
        """
        fmap = []

        # 1D音声を2D表現に変換
        b, c, t = x.shape
        if t % self.period != 0:  # 周期で割り切れない場合はパディング
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        # [batch, channels, time] → [batch, channels, time//period, period]
        x = x.view(b, c, t // self.period, self.period)

        # 各畳み込み層を通過
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)  # Feature Matching用に特徴マップを保存

        # 最終層で真偽スコアを出力
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    """
    Scale-based Discriminator (スケールベース識別器)

    1D畳み込みを使用して、異なる時間スケールでの音声パターンを識別します。
    Multi-Period Discriminatorの一部として使用されます。

    アーキテクチャ:
        - 6層の1D畳み込み層
        - Grouped Convolutionを使用 (計算効率化)
        - チャンネル数: 1→16→64→256→1024→1024→1024
        - ストライド4で時間解像度を段階的に削減

    特徴:
        - 異なる時間スケールのパターンを捉える
        - 各層の特徴マップを保存 (Feature Matching Loss用)
        - Weight NormまたはSpectral Normを使用可能
    """

    def __init__(self, use_spectral_norm: bool = False) -> None:
        """
        Scale-based Discriminatorを初期化します。

        Args:
            use_spectral_norm (bool): Spectral Normalizationを使用するか
                Falseの場合はWeight Normを使用. Defaults to False.
        """
        super(DiscriminatorS, self).__init__()
        # 正規化関数を選択 (Weight Norm or Spectral Norm)
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        # 畳み込み層のリスト
        # Grouped Convolutionで計算効率を向上
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        # 最終出力層: 1024→1チャンネル (真偽スコア)
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        順伝播処理

        1D畳み込みを使用して音声の真偽を判定します。

        Args:
            x (torch.Tensor): 入力音声波形
                Shape: [batch, 1, time]

        Returns:
            tuple: 以下の要素を含むタプル
                - 判定スコア (torch.Tensor): 真偽判定結果
                    Shape: [batch, *] (flatten後)
                - 特徴マップリスト (list[torch.Tensor]): 各層の出力
                    Feature Matching Loss計算に使用
        """
        fmap = []

        # 各畳み込み層を通過
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)  # Feature Matching用に特徴マップを保存

        # 最終層で真偽スコアを出力
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-Period Discriminator (複数周期識別器)

    複数の異なる周期を持つDiscriminatorPと1つのDiscriminatorSを
    組み合わせて、様々な時間スケールでの音声パターンを識別します。

    構成:
        - 1つのScale-based Discriminator (DiscriminatorS)
        - 5つのPeriod-based Discriminator (周期: 2, 3, 5, 7, 11)

    利点:
        - 異なる周期パターンを同時に識別
        - 多様な時間スケールでの音声特徴を捉える
        - HiFi-GANの識別器アーキテクチャを採用
    """

    def __init__(self, use_spectral_norm: bool = False) -> None:
        """
        Multi-Period Discriminatorを初期化します。

        Args:
            use_spectral_norm (bool): Spectral Normalizationを使用するか
                Falseの場合はWeight Normを使用. Defaults to False.
        """
        super(MultiPeriodDiscriminator, self).__init__()
        # 周期のリスト: 素数を使用することで多様なパターンを捉える
        periods = [2, 3, 5, 7, 11]

        # Scale-based Discriminatorを追加
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        # 各周期でPeriod-based Discriminatorを追加
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm)
            for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        順伝播処理

        本物と生成音声の両方を全ての識別器に通し、判定結果と特徴マップを取得します。

        Args:
            y (torch.Tensor): 本物の音声波形 (Real)
                Shape: [batch, 1, time]
            y_hat (torch.Tensor): 生成された音声波形 (Generated)
                Shape: [batch, 1, time]

        Returns:
            tuple: 以下の4つのリストを含むタプル
                - y_d_rs (list[torch.Tensor]): 本物音声の判定スコアのリスト
                    各識別器からの出力
                - y_d_gs (list[torch.Tensor]): 生成音声の判定スコアのリスト
                    各識別器からの出力
                - fmap_rs (list[torch.Tensor]): 本物音声の特徴マップのリスト
                    Feature Matching Loss計算に使用
                - fmap_gs (list[torch.Tensor]): 生成音声の特徴マップのリスト
                    Feature Matching Loss計算に使用
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # 全ての識別器で本物と生成音声を判定
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)  # 本物音声
            y_d_g, fmap_g = d(y_hat)  # 生成音声
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DurationDiscriminator(nn.Module):
    """
    Duration Discriminator (持続時間識別器)

    テキストエンコーダの出力と持続時間情報を使用して、
    持続時間予測の真偽を判定します。

    アーキテクチャ:
        - 2層の1D畳み込み層 (正規化とドロップアウト付き)
        - 双方向LSTM層
        - 最終出力層 (シグモイド活性化)

    用途:
        - 持続時間予測器の学習を改善
        - より自然な発話リズムの生成に寄与
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        """
        Duration Discriminatorを初期化します。

        Args:
            in_channels (int): 入力チャンネル数
            filter_channels (int): フィルターチャンネル数
            kernel_size (int): 畳み込みカーネルサイズ
            p_dropout (float): ドロップアウト率
            gin_channels (int): 話者条件付け用のチャンネル数.
                Defaults to 0.
        """
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)

        # 1層目の畳み込み
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)

        # 2層目の畳み込み
        self.conv_2 = nn.Conv1d(
            filter_channels,
            filter_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.norm_2 = LayerNorm(filter_channels)

        # LSTM block: 持続時間情報を処理
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)
        self.LSTM = nn.LSTM(
            2 * filter_channels,
            filter_channels,
            batch_first=True,
            bidirectional=True,
        )

        # 話者条件付け用の層
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 最終出力層: 真偽確率を出力 (0-1)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * filter_channels, 1), nn.Sigmoid()
        )

    def forward_probability(
        self,
        x: torch.Tensor,
        dur: torch.Tensor,
    ) -> torch.Tensor:
        """
        持続時間条件付き確率を計算します。

        Args:
            x (torch.Tensor): テキストエンコーダからの特徴量
                Shape: [batch, filter_channels, time]
            dur (torch.Tensor): 持続時間テンソル
                Shape: [batch, 1, time]

        Returns:
            torch.Tensor: 持続時間の真偽確率
                Shape: [batch, time, 1]
        """
        # 持続時間を射影
        dur = self.dur_proj(dur)
        # テキスト特徴量と持続時間を結合
        x = torch.cat([x, dur], dim=1)
        x = x.transpose(1, 2)  # [batch, time, channels]
        # LSTMで系列情報を処理
        x, _ = self.LSTM(x)
        # 最終的な確率を出力
        output_prob = self.output_layer(x)  # [batch, time, 1]
        return output_prob

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        dur_r: torch.Tensor,
        dur_hat: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        順伝播処理

        本物の持続時間と予測された持続時間の両方を評価します。

        Args:
            x (torch.Tensor): テキストエンコーダからの特徴量
                Shape: [batch, in_channels, time]
            x_mask (torch.Tensor): テキストマスク
                Shape: [batch, 1, time]
            dur_r (torch.Tensor): 正解の持続時間 (Real)
                Shape: [batch, 1, time]
            dur_hat (torch.Tensor): 予測された持続時間 (Generated)
                Shape: [batch, 1, time]
            g (Optional[torch.Tensor]): 話者埋め込み
                Shape: [batch, gin_channels, 1]. Defaults to None.

        Returns:
            list[torch.Tensor]: 2つの確率テンソルのリスト
                - output_probs[0]: 正解持続時間の真偽確率
                - output_probs[1]: 予測持続時間の真偽確率
                Shape: [batch, time, 1]
        """
        # 勾配を切断して特徴量を取得
        x = torch.detach(x)

        # 話者条件付け
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        # 1層目の畳み込み
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        # 2層目の畳み込み
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        # 正解と予測の両方の持続時間を評価
        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, dur)
            output_probs.append(output_prob)

        return output_probs
