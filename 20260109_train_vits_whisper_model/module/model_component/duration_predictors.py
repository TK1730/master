import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.model_component.modules import Flip
from module.model_component.normalization import LayerNorm
from module.model_component.modules import (
    DDSConv,
    ElementwiseAffine,
    Log,
    ConvFlow,
)


class StochasticDurationPredictor(nn.Module):
    """
    Normalizing Flowsを使用した確率的Duration予測器

    このモジュールは、Normalizing Flowsを用いた変分推論アプローチで
    duration予測を実装します。テキストに条件付けられた事前分布 p(z|x) と、
    テキストと目標durationの両方に条件付けられた事後分布 q(z|x,w) を学習します。

    モデルはELBOを最大化することで学習されます:
        ELBO = E_q[log p(w|z)] - KL[q(z|x,w) || p(z|x)]

    推論時には、学習された事前分布 p(z|x) からdurationがサンプリングされます。

    __init__の引数:
        in_channels: 入力特徴量のチャネル数
        filter_channels: フィルタチャネル数(in_channelsで上書きされます)
        kernel_size: 畳み込みカーネルサイズ
        p_dropout: ドロップアウト確率
        n_flows: 事前分布のFlow層数(デフォルト: 4)
        gin_channels: 話者埋め込みチャネル数(単一話者の場合は0)
    """
    @staticmethod
    def _create_flow_modules(
        n_flows: int,
        filter_channels: int,
        kernel_size: int,
    ) -> nn.ModuleList:
        """
        Normalizing Flow modules を作成するヘルパーメソッド

        Args:
            n_flows (int): Flow層の数
            filter_channels (int): フィルタチャネル数
            kernel_size (int): カーネルサイズ

        Returns:
            nn.ModuleList: ElementwiseAffine -> (ConvFlow -> Flip) * n_flows
        """
        flows = nn.ModuleList()
        flows.append(ElementwiseAffine(2))
        for _ in range(n_flows):
            flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            flows.append(Flip())
        return flows

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        """
        確率的Duration予測器の初期化

        Args:
            in_channels: 入力特徴量のチャネル数
            filter_channels: フィルタチャネル数
                (注意: これはin_channelsで上書きされます)
            kernel_size: 畳み込みカーネルサイズ
            p_dropout: ドロップアウト確率
            n_flows: Normalizing Flow層の数(デフォルト: 4)
            gin_channels: 話者埋め込みのチャネル数
                (単一話者モデルの場合は0)
        """
        super().__init__()
        # NOTE: filter_channelsは将来のバージョンで削除予定
        filter_channels = in_channels

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 事前分布のflows: p(z|x)
        self.log_flow = Log()
        self.flows = self._create_flow_modules(
            n_flows, filter_channels, kernel_size
        )

        # 事後分布のエンコーダ
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_convs = DDSConv(
            channels=filter_channels,
            kernel_size=kernel_size,
            n_layers=3,
            p_dropout=p_dropout,
        )
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 事後分布のflows: q(z|x,w)
        self.post_flows = self._create_flow_modules(
            n_flows=n_flows,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
        )

        # テキストエンコーダ
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.convs = DDSConv(
            channels=filter_channels,
            kernel_size=kernel_size,
            n_layers=3,
            p_dropout=p_dropout,
        )
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)

        # 話者条件付け（多話者モデル用）
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def _encode_text(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        テキスト特徴量をエンコード

        Args:
            x: 入力特徴量 (B, C, T)
            x_mask: マスク (B, 1, T)
            g: 話者埋め込み (B, gin_channels) またはNone

        Returns:
            エンコードされたテキスト特徴量 (B, C, T)
        """
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        return x

    def _compute_posterior_logq(
        self,
        w: torch.Tensor,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        事後分布 q(z|x,w) を計算

        Args:
            w: 目標duration (B, 1, T)
            x: テキストエンコーディング (B, C, T)
            x_mask: マスク (B, 1, T)

        Returns:
            (z0, z1, e_q, logdet_tot_q) のタプル
        """
        # duration条件のエンコーディング
        h_w = self.post_pre(w)
        h_w = self.post_convs(h_w, x_mask)
        h_w = self.post_proj(h_w) * x_mask

        # ノイズをサンプリング
        e_q = (
            torch.randn(
                w.size(0), 2, w.size(2)
            ).to(device=x.device, dtype=x.dtype)
            * x_mask
        )
        z_q = e_q

        # 事後分布flowsを適用
        logdet_tot_q = 0.0
        for flow in self.post_flows:
            z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
            logdet_tot_q += logdet_q

        # duration成分とノイズ成分に分割
        z_u, z1 = torch.split(z_q, [1, 1], 1)
        u = torch.sigmoid(z_u) * x_mask
        z0 = (w - u) * x_mask

        # sigmoidのヤコビアンを追加
        logdet_tot_q += torch.sum(
            (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
        )

        # log q(z|x,w) を計算
        logq = (
            torch.sum(
                -0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask,
                [1, 2]
            ) - logdet_tot_q
        )

        return z0, z1, e_q, logq

    def _compute_prior_nll(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        事前分布 p(z|x) の下での負の対数尤度を計算

        Args:
            z0: duration成分 (B, 1, T)
            z1: ノイズ成分 (B, 1, T)
            x: テキストエンコーディング (B, C, T)
            x_mask: マスク (B, 1, T)

        Returns:
            負の対数尤度 (B,)
        """
        logdet_tot = 0.0

        # Log flowを適用（durationが正であることを保証）
        z0, logdet = self.log_flow(z0, x_mask)
        logdet_tot += logdet

        # z0とz1を結合
        z = torch.cat([z0, z1], 1)

        # 事前分布flowsを適用
        for flow in self.flows:
            z, logdet = flow(z, x_mask, g=x, reverse=False)
            logdet_tot += logdet

        # ガウス事前分布の下でのNLLを計算
        nll = (
            torch.sum(
                0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]
            ) - logdet_tot
        )
        return nll

    def _sample_from_prior(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        """
        事前分布からdurationをサンプリング(推論モード)

        Args:
            x: テキストエンコーディング (B, C, T)
            x_mask: マスク (B, 1, T)
            noise_scale: サンプリング時のノイズスケール

        Returns:
            対数duration (B, 1, T)
        """
        # Flowsを逆順に（最後のFlipを削除してElementwiseAffineを保持）
        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]

        # 標準ガウス分布からサンプリング
        z = (
            torch.randn(
                x.size(0), 2, x.size(2)
            ).to(device=x.device, dtype=x.dtype)
            * noise_scale
        )

        # Flowsを逆方向に適用
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)

        # duration成分を抽出
        z0, _ = torch.split(z, [1, 1], 1)
        return z0

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        確率的duration予測の順伝播

        学習モード(reverse=False): ELBOロスを計算
        推論モード(reverse=True): 学習された事前分布からdurationをサンプリング

        Args:
            x: 入力特徴量 (B, C, T)
            x_mask: マスクテンソル (B, 1, T)
            w: 目標duration (B, 1, T)、学習時に必要
            g: 話者埋め込み (B, gin_channels)、オプション
            reverse: Falseの場合はロスを計算、Trueの場合はdurationをサンプリング
            noise_scale: サンプリング時のノイズスケール

        Returns:
            学習時: 負のELBO (B,)
            推論時: 対数duration (B, 1, T)
        """
        # テキスト特徴量をエンコード
        x = self._encode_text(x, x_mask, g)

        if not reverse:
            # 学習モード: ELBOを計算
            assert w is not None, (
                "Duration w must be provided in training mode"
            )

            # 事後分布 q(z|x,w) を計算
            z0, z1, e_q, logq = self._compute_posterior_logq(w, x, x_mask)

            # -log p(z|x) を計算
            nll = self._compute_prior_nll(z0, z1, x, x_mask)

            # ELBOを返す: -log p(z|x) + log q(z|x,w)
            return nll + logq
        else:
            # 推論モード: 事前分布 p(z|x) からサンプリング
            return self._sample_from_prior(x, x_mask, noise_scale)


class ConvBlock(nn.Module):
    """
    畳み込みブロック: Conv1d -> ReLU -> LayerNorm -> Dropout

    duration predictorで使用される基本的な畳み込みブロック。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        """
        Args:
            in_channels (int): 入力チャネル数
            out_channels (int): 出力チャネル数
            kernel_size (int): カーネルサイズ
            p_dropout (float): ドロップアウト率
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = LayerNorm(out_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)

        Returns:
            torch.Tensor: 出力テンソル (B, C, T)
        """
        x = self.conv(x * x_mask)
        x = torch.relu(x)
        x = self.norm(x)
        x = self.drop(x)
        return x


class DurationPredictor(nn.Module):
    """
    決定的Duration予測器

    このモジュールは、畳み込みニューラルネットワークを使用して
    音素ごとのduration(継続時間)を予測します。
    StochasticDurationPredictorとは異なり、入力特徴量が与えられると
    常に同じduration予測値を返す決定的なモデルです。

    アーキテクチャ:
        1. 話者条件付け（オプション）: 話者埋め込みを入力特徴量に加算
        2. 畳み込みブロックx2: それぞれがConv1d → ReLU → LayerNorm → Dropoutを含む
        3. 射影層: filter_channelsから1チャネルへの1x1畳み込み
        4. 出力: 予測されたduration値(対数スケールまたは線形スケール)

    __init__の引数:
        in_channels: 入力特徴量のチャネル数
        filter_channels: 畳み込みフィルタのチャネル数
        kernel_size: 畳み込みカーネルサイズ
        p_dropout: ドロップアウト確率
        gin_channels: 話者埋め込みチャネル数(単一話者の場合は0)
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
        決定的Duration予測器の初期化

        Args:
            in_channels: 入力特徴量のチャネル数（通常はテキストエンコーダの出力次元）
            filter_channels: 畳み込みレイヤーのフィルタチャネル数
            kernel_size: 畳み込みカーネルサイズ（奇数を推奨）
            p_dropout: ドロップアウト確率(0.0~1.0)
            gin_channels: 話者埋め込みのチャネル数
                         (単一話者モデルの場合は0、多話者モデルの場合は > 0)
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 畳み込みブロックのスタック
        # ブロック1: in_channels → filter_channels
        # ブロック2: filter_channels → filter_channels
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                in_channels, filter_channels, kernel_size, p_dropout
            ),
            ConvBlock(
                filter_channels, filter_channels, kernel_size, p_dropout
            ),
        ])

        # Duration予測のための射影層（1チャネル出力）
        self.proj = nn.Conv1d(filter_channels, 1)

        # 話者条件付け層（多話者モデルの場合）
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Duration予測の順伝播

        Args:
            x: 入力特徴量テンソル (B, in_channels, T)
                B: バッチサイズ
                in_channels: 入力チャネル数
                T: 時間ステップ数（音素シーケンス長）
            x_mask: マスクテンソル (B, 1, T)
                パディング位置を0、有効位置を1とするバイナリマスク
            g: 話者埋め込みテンソル (B, gin_channels, 1) またはNone
                多話者モデルの場合に使用

        Returns:
            torch.Tensor: 予測されたduration (B, 1, T)
                各音素に対する予測duration値
        """
        # 勾配計算グラフから入力を切り離す
        # （Duration予測器はテキストエンコーダからの勾配を受け取らない）
        x = torch.detach(x)
        # 話者条件付け（多話者モデルの場合）
        if g is not None:
            g = torch.detach(g)
            # 話者埋め込みを入力特徴量に加算
            x = x + self.cond(g)

        # 畳み込みブロックを順次適用
        for conv_block in self.conv_blocks:
            x = conv_block(x, x_mask)

        # 射影層でdurationを予測（1チャネルに変換）
        x = self.proj(x * x_mask)

        # マスクを適用して無効な位置をゼロに
        return x * x_mask
