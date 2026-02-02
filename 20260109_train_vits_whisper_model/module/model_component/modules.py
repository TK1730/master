import math
from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from utils.model import (
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    init_weights,
)
from module.model_component.normalization import LayerNorm
from module.model_component.transformer import FFT
from utils.transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ) -> None:
        """
        WaveNet (WN) モジュール

        Dilated Causal Convolutionを使用した生成モデル。
        Gated Activation (tanh * sigmoid)とResidual/Skip接続を特徴とします。
        オプションで話者埋め込みによる条件付けをサポートします。

        Args:
            hidden_channels (int): 隠れ層のチャネル数
            kernel_size (int): 畳み込みカーネルサイズ(奇数である必要があります)
            dilation_rate (int): 各層のdilationの倍率
                (i層目のdilation = dilation_rate^i)
            n_layers (int): WaveNet層の数
            gin_channels (int, optional): 話者埋め込みのチャネル数
                (0の場合は条件付けなし) (default: 0)
            p_dropout (float, optional): ドロップアウト率 (default: 0)
        """
        super(WN, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        # モジュールリストの初期化
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        # 条件付けレイヤーの初期化(話者埋め込みがある場合)
        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = weight_norm(cond_layer, name="weight")

        # WaveNet層の構築
        for i in range(n_layers):
            # Dilationを指数的に増加
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)

            # Gated Activation用の入力層(2倍のチャネル数)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # Residual/Skip接続用の出力層
            # 最終層はスキップ接続のみなのでチャネル数が半分
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(
                hidden_channels, res_skip_channels, 1
            )
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def _get_conditional_input(
        self, g: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        指定した層の条件付け入力を取得

        Args:
            g (torch.Tensor): 変換済み条件付け入力 (B, 2*hidden_channels*n_layers, T)
            layer_idx (int): 層のインデックス

        Returns:
            torch.Tensor: 該当層の条件付け入力 (B, 2*hidden_channels, T)
        """
        cond_offset = layer_idx * 2 * self.hidden_channels
        return g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        WaveNetの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, hidden_channels, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)
            g (Optional[torch.Tensor], optional): 話者埋め込みテンソル
                (B, gin_channels, T)
            **kwargs (Any): その他の引数

        Returns:
            torch.Tensor: 出力テンソル (B, hidden_channels, T)
        """
        # スキップ接続の累積用出力を初期化
        output = torch.zeros_like(x)

        # 条件付け入力の前処理
        if g is not None:
            g = self.cond_layer(g)

        # 各WaveNet層の処理
        for i in range(self.n_layers):
            # 入力層を通す(Gated Activation用に2倍のチャネル)
            x_in = self.in_layers[i](x)

            # 条件付け入力の取得
            if g is not None:
                g_l = self._get_conditional_input(g, i)
            else:
                g_l = torch.zeros_like(x_in)

            # Gated Activation: tanh(x_in + g) * sigmoid(x_in + g)
            acts = fused_add_tanh_sigmoid_multiply(
                x_in, g_l, [self.hidden_channels]
            )
            acts = self.drop(acts)

            # Residual/Skip接続の処理
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                # 最終層以外: ResidualとSkipの両方
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask  # Residual接続
                # Skip接続
                output = (
                    output + res_skip_acts[:, self.hidden_channels:, :]
                )
            else:
                # 最終層: Skipのみ
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        """
        すべてのレイヤーからWeight Normalizationを削除

        推論時にWeight Normalizationのオーバーヘッドを削減するために使用します。
        """
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class DDSConv(nn.Module):
    """
    Dilated and Depth-Separable Convolution
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ) -> None:
        """
        Args:
            channels (int): 入力チャネル数
            kernel_size (int): カーネルサイズ
            n_layers (int): 層数
            p_dropout (float, optional): ドロップアウト率. Defaults to 0.0.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()  # 膨張-深さ方向分離畳み込みの層
        self.linears = nn.ModuleList()  # 線形変換の層
        self.norms_1 = nn.ModuleList()  # 正規化の層
        self.norms_2 = nn.ModuleList()  # 正規化の層
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = get_padding(kernel_size, dilation)
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.linears.append(nn.Linear(channels, channels))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDSConvの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)
            g (Optional[torch.Tensor], optional): 条件付け入力
                (B, channels, T)

        Returns:
            torch.Tensor: 出力テンソル (B, channels, T)
        """
        # 条件付け入力を加算
        if g is not None:
            x = x + g

        # 各層を順次適用
        for i in range(self.n_layers):
            # Depth-separable convolution (dilated)
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)

            # Pointwise convolution (1x1畳み込み相当)
            y = self.linears[i](y.mT).mT
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)

            # Residual接続
            x = x + y
        return x * x_mask


class Log(nn.Module):
    """
    対数変換レイヤー (Normalizing Flow用)

    正規化フローの一部として、対数変換とその逆変換を提供します。
    log-determinant of Jacobianを計算して返します。
    """

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs: Any,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        対数変換の順伝播または逆伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)
            reverse (bool, optional): 逆変換を行うか (default: False)
            **kwargs (Any): その他の引数

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                reverse=False: (y, logdet)
                reverse=True: x
        """
        if not reverse:
            # 順変換: y = log(x)
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            # log-determinant: -sum(log(x)) = -sum(y)
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            # 逆変換: x = exp(y)
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    """
    チャネル反転レイヤー (Normalizing Flow用)

    正規化フローの一部として、チャネルの順序を反転します。
    この操作はJacobianが単位行列なので、log-determinantは0です。
    """

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        reverse: bool = False,
        **kwargs: Any,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        チャネル反転の順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            reverse (bool, optional): 逆変換を行うか (default: False)
            *args, **kwargs: その他の引数(無視)

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                reverse=False: (x反転, logdet=0)
                reverse=True: x反転
        """
        # チャネル次元（dim=1）を反転
        x = torch.flip(x, [1])
        if not reverse:
            # log-determinantは0（単位行列のJacobian）
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    """
    要素ごとのアフィン変換 (Normalizing Flow用)

    各チャネルに対して y = scale * x + shift の変換を適用します。
    scaleとshiftは学習可能なパラメータです。
    """

    def __init__(self, channels: int) -> None:
        """
        Args:
            channels (int): チャネル数
        """
        super().__init__()
        self.channels = channels
        # shiftパラメータ
        self.m = nn.Parameter(torch.zeros(channels, 1))
        # log(scale)パラメータ
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs: Any,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        アフィン変換の順伝播または逆伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)
            reverse (bool, optional): 逆変換を行うか (default: False)
            **kwargs (Any): その他の引数

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                reverse=False: (y, logdet)
                reverse=True: x
        """
        if not reverse:
            # 順変換: y = m + exp(logs) * x
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            # log-determinant: sum(logs)
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            # 逆変換: x = (y - m) * exp(-logs)
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ConvFlow(nn.Module):
    """
    Coupling Layer with Rational Quadratic Spline (Normalizing Flow用)

    正規化フローのカップリング層。
    入力を2つに分割し、片方を条件としてもう片方に
    Rational Quadratic Spline変換を適用します。
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ) -> None:
        """
        Args:
            in_channels (int): 入力チャネル数
            filter_channels (int): フィルターチャネル数
            kernel_size (int): カーネルサイズ
            n_layers (int): DDSConvの層数
            num_bins (int, optional): Splineのビン数 (default: 10)
            tail_bound (float, optional): Splineの端の境界 (default: 5.0)
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        # 変換パラメータ生成用のネットワーク
        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(
            filter_channels, kernel_size, n_layers, p_dropout=0.0
        )
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        # パラメータを0で初期化(恒等変換から開始)
        self.proj.weight.data.zero_()
        assert self.proj.bias is not None
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        ConvFlowの順伝播または逆伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, in_channels, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)
            g (Optional[torch.Tensor], optional): 条件付け入力
            reverse (bool, optional): 逆変換を行うか (default: False)

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                reverse=False: (x, logdet)
                reverse=True: x
        """
        # 入力を2つに分割 (coupling layer)
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # x0を条件として変換パラメータを生成
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        # Splineパラメータをチャネル次元から時間次元に変換
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(
            0, 1, 3, 2
        )  # [b, c, t, 3*num_bins-1]

        # Splineパラメータを抽出し正規化
        denom = math.sqrt(self.filter_channels)
        unnormalized_widths = h[..., :self.num_bins] / denom
        unnormalized_heights = (
            h[..., self.num_bins:2 * self.num_bins] / denom
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins:]

        # Rational Quadratic Spline変換をx1に適用
        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        # x0とx1を連結
        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(nn.Module):
    """
    Residual Coupling Layer (正規化フローのアフィン結合層)

    正規化フロー(Normalizing Flow)における可逆変換層の一つです。
    入力をチャネル方向で2つに分割し、一方のチャネル(x0)を使って
    他方のチャネル(x1)をアフィン変換します。

    変換の仕組み:
        順方向: x1_new = x1 * exp(logs) + m
        逆方向: x1_new = (x1 - m) * exp(-logs)
        ※ m (平均) と logs (対数スケール) は x0 から計算

    この変換は可逆であり、対数行列式(logdet)を計算できるため、
    VAE(変分オートエンコーダ)などの生成モデルで確率密度を扱う際に使用されます。

    話者条件付けをサポートしており、gin_channelsを0以外に設定することで
    多話者モデルに対応できます。
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
        wn_sharing_parameter: Optional[nn.Module] = None,
    ) -> None:
        """
        Residual Coupling Layerの初期化

        Args:
            channels (int): 入力チャネル数(2で割り切れる必要がある)
            hidden_channels (int): WaveNetエンコーダの隠れ層チャネル数
            kernel_size (int): WaveNetの畳み込みカーネルサイズ
            dilation_rate (int): WaveNetの膨張率(dilation rate)
            n_layers (int): WaveNetの層数
            p_dropout (float, optional): ドロップアウト率。デフォルトは0。
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                0の場合は単一話者、0以外の場合は多話者モデル。デフォルトは0。
            mean_only (bool, optional): Trueの場合、スケーリング(logs)を省略し、
                平均(m)のみのシフト変換を行う。デフォルトはFalse。
            wn_sharing_parameter (Optional[nn.Module], optional): 複数の層で
                WaveNetパラメータを共有する場合に指定。Noneの場合は新規WNを
                作成。デフォルトはNone。
        """
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        # 前処理: 分割したチャネルをhidden_channelsに変換
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # WaveNetエンコーダ: x0から変換パラメータを計算
        # wn_sharing_parameterが指定されている場合はそれを使用、
        # そうでない場合は新規WNを作成
        self.enc = (
            WN(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                p_dropout=p_dropout,
                gin_channels=gin_channels,
            )
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )

        # 後処理: hidden_channelsから変換パラメータ(m, logs)に射影
        # mean_only=Trueの場合はmのみ(half_channels)、Falseの場合はm+logs(half_channels*2)
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )
        # 後処理層の重みとバイアスを0で初期化(恒等変換から学習を開始)
        self.post.weight.data.zero_()
        assert self.post.bias is not None
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Residual Coupling Layerの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
                - B: バッチサイズ
                - channels: チャネル数(2で割り切れる)
                - T: 時系列長
            x_mask (torch.Tensor): パディングマスク (B, 1, T)
                有効な位置を1、パディング位置を0とするマスク
            g (Optional[torch.Tensor], optional): 話者埋め込みベクトル
                (B, gin_channels)。gin_channels=0の場合は不要。
                デフォルトはNone。
            reverse (bool, optional): 逆変換を行うかどうか。
                False: 順方向変換(x -> z)
                True: 逆方向変換(z -> x)
                デフォルトはFalse。

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                reverse=Falseの場合: (x, logdet)
                    - x: 変換後のテンソル (B, channels, T)
                    - logdet: 対数行列式 (B,)
                reverse=Trueの場合: x
                    - x: 逆変換後のテンソル (B, channels, T)
        """
        # 1. チャネルを2つに分割
        # x0: 変換パラメータを計算するための入力
        # x1: 実際に変換される対象
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # 2. x0から変換パラメータ(m, logs)を計算
        # 前処理: x0を隠れ層の次元に変換
        h = self.pre(x0) * x_mask  # [B, hidden_channels, T]
        # WaveNetエンコーダで特徴を抽出(話者埋め込みgで条件付け可能)
        h = self.enc(h, x_mask, g=g)  # [B, hidden_channels, T]
        # 後処理: 変換パラメータに射影
        stats = self.post(h) * x_mask  # [B, half_channels*(1 or 2), T]

        # 3. 変換パラメータを分割
        if not self.mean_only:
            # 通常モード: 平均(m)と対数スケール(logs)の両方を使用
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            # mean_onlyモード: 平均(m)のみ使用、スケーリングなし
            m = stats
            logs = torch.zeros_like(m)

        # 4. アフィン変換の適用
        if not reverse:
            # 順方向変換: x1 -> (x1 * exp(logs) + m)
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)  # x0とx1を再結合
            # ヤコビアンの対数行列式を計算(確率密度の変換に必要)
            logdet = torch.sum(logs, [1, 2])  # [B]
            return x, logdet
        else:
            # 逆方向変換: (x1 - m) * exp(-logs)
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)  # x0とx1を再結合
            return x


class TransformerCouplingLayer(nn.Module):
    """
    Transformer Coupling Layer (Transformerを使用した正規化フローの結合層)

    ResidualCouplingLayerと同様の可逆変換層ですが、WaveNetの代わりに
    FFT(Feed-Forward Transformer)を使用しています。これにより、より長い文脈を
    捉えることができます。

    変換の仕組み:
        順方向: x1_new = x1 * exp(logs) + m
        逆方向: x1_new = (x1 - m) * exp(-logs)
        ※ m (平均) と logs (対数スケール) は x0 をFFTで処理して計算

    ResidualCouplingLayerとの違い:
        - WaveNetの代わりにFFTを使用
        - FFTはisflow=Trueで初期化され、話者条件付けをサポート
        - パラメータ共有(wn_sharing_parameter)に対応
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float = 0,
        filter_channels: int = 0,
        mean_only: bool = False,
        wn_sharing_parameter: Optional[nn.Module] = None,
        gin_channels: int = 0,
    ) -> None:
        """
        Transformer Coupling Layerの初期化

        Args:
            channels (int): 入力チャネル数(2で割り切れる必要がある)
            hidden_channels (int): FFTエンコーダの隠れ層チャネル数
            kernel_size (int): FFT内のFFN層の畳み込みカーネルサイズ
            n_layers (int): FFTのTransformer層数
            n_heads (int): FFTのマルチヘッドアテンションのヘッド数
            p_dropout (float, optional): ドロップアウト率。デフォルトは0。
            filter_channels (int, optional): FFT内のFFN層の中間チャネル数。
                デフォルトは0。
            mean_only (bool, optional): Trueの場合、スケーリング(logs)を省略し、
                平均(m)のみのシフト変換を行う。デフォルトはFalse。
            wn_sharing_parameter (Optional[nn.Module], optional): 複数の層で
                FFTパラメータを共有する場合に指定。Noneの場合は新規FFTを
                作成。デフォルトはNone。
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                0の場合は単一話者、0以外の場合は多話者モデル。
                デフォルトは0。
        """
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        # 前処理: 分割したチャネルをhidden_channelsに変換
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # FFTエンコーダ: x0から変換パラメータを計算
        # wn_sharing_parameterが指定されている場合はそれを使用、
        # そうでない場合は新規FFTを作成
        self.enc = (
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
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )

        # 後処理: hidden_channelsから変換パラメータ(m, logs)に射影
        # mean_only=Trueの場合はmのみ(half_channels)、
        # Falseの場合はm+logs(half_channels*2)
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1
        )
        # 後処理層の重みとバイアスを0で初期化(恒等変換から学習を開始)
        self.post.weight.data.zero_()
        assert self.post.bias is not None
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Transformer Coupling Layerの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
                - B: バッチサイズ
                - channels: チャネル数(2で割り切れる)
                - T: 時系列長
            x_mask (torch.Tensor): パディングマスク (B, 1, T)
                有効な位置を1、パディング位置を0とするマスク
            g (Optional[torch.Tensor], optional): 話者埋め込みベクトル
                (B, gin_channels)。FFTのisflow=Trueで話者条件付けを有効化。
                デフォルトはNone。
            reverse (bool, optional): 逆変換を行うかどうか。
                False: 順方向変換(x -> z)
                True: 逆方向変換(z -> x)
                デフォルトはFalse。

        Returns:
            Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                reverse=Falseの場合: (x, logdet)
                    - x: 変換後のテンソル (B, channels, T)
                    - logdet: 対数行列式 (B,)
                reverse=Trueの場合: x
                    - x: 逆変換後のテンソル (B, channels, T)
        """
        # 1. チャネルを2つに分割
        # x0: 変換パラメータを計算するための入力
        # x1: 実際に変換される対象
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # 2. x0から変換パラメータ(m, logs)を計算
        # 前処理: x0を隠れ層の次元に変換
        h = self.pre(x0) * x_mask  # [B, hidden_channels, T]
        # FFTエンコーダで特徴を抽出
        # FFTはisflow=Trueで初期化されているため、話者埋め込みgで条件付け可能
        h = self.enc(h, x_mask, g=g)  # [B, hidden_channels, T]
        # 後処理: 変換パラメータに射影
        stats = self.post(h) * x_mask  # [B, half_channels*(1 or 2), T]

        # 3. 変換パラメータを分割
        if not self.mean_only:
            # 通常モード: 平均(m)と対数スケール(logs)の両方を使用
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            # mean_onlyモード: 平均(m)のみ使用、スケーリングなし
            m = stats
            logs = torch.zeros_like(m)

        # 4. アフィン変換の適用
        if not reverse:
            # 順方向変換: x1 -> (x1 * exp(logs) + m)
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)  # x0とx1を再結合
            # ヤコビアンの対数行列式を計算(確率密度の変換に必要)
            logdet = torch.sum(logs, [1, 2])  # [B]
            return x, logdet
        else:
            # 逆方向変換: (x1 - m) * exp(-logs)
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)  # x0とx1を再結合
            return x


class ResBlock1(torch.nn.Module):
    """
    Residual Block Type 1 (HiFi-GAN用)

    3つの異なるdilationを持つ畳み込み層と、
    それぞれに対応dilation=1の畳み込み層を2層持ちます。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        """
        Args:
            channels (int): 入力/出力チャネル数
            kernel_size (int, optional): カーネルサイズ (default: 3)
            dilation (tuple[int, int, int], optional):
                3つのdilation率 (default: (1, 3, 5))
        """
        super(ResBlock1, self).__init__()

        # 異なるdilationを持つ畳み込み層
        self.convs1 = self._create_conv_layers(
            channels, kernel_size, dilation
        )
        self.convs1.apply(init_weights)

        # dilation=1の畳み込み層
        self.convs2 = self._create_conv_layers(
            channels, kernel_size, (1, 1, 1)
        )
        self.convs2.apply(init_weights)

    def _create_conv_layers(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, int, int],
    ) -> nn.ModuleList:
        """
        Weight Normalizationを適用した畳み込み層のリストを生成

        Args:
            channels (int): 入力/出力チャネル数
            kernel_size (int): カーネルサイズ
            dilations (tuple[int, int, int]): 3つのdilation率

        Returns:
            nn.ModuleList: 畳み込み層のリスト
        """
        return nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilations
        ])

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ResBlock1の順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
            x_mask (Optional[torch.Tensor], optional):
                マスクテンソル (B, 1, T)

        Returns:
            torch.Tensor: 出力テンソル (B, channels, T)
        """
        # 3つの(conv1, conv2)ペアを順次適用
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            # dilationありの畳み込み
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            # dilation=1の畳み込み
            xt = c2(xt)
            # Residual接続
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """
        Weight Normalizationを削除
        """
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(torch.nn.Module):
    """
    Residual Block Type 2 (HiFi-GAN用)

    2つの異なるdilationを持つ畳み込み層を2層持ちます。
    ResBlock1よりもシンプルな構造です。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
    ) -> None:
        """
        Args:
            channels (int): 入力/出力チャネル数
            kernel_size (int, optional): カーネルサイズ (default: 3)
            dilation (tuple[int, int], optional):
                2つのdilation率 (default: (1, 3))
        """
        super(ResBlock2, self).__init__()

        # 畳み込み層のリストを生成
        self.convs = self._create_conv_layers(channels, kernel_size, dilation)
        self.convs.apply(init_weights)

    def _create_conv_layers(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, int],
    ) -> nn.ModuleList:
        """
        Weight Normalizationを適用した畳み込み層のリストを生成

        Args:
            channels (int): 入力/出力チャネル数
            kernel_size (int): カーネルサイズ
            dilations (tuple[int, int]): dilation率

        Returns:
            nn.ModuleList: 畳み込み層のリスト
        """
        return nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilations
        ])

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ResBlock2の順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, channels, T)
            x_mask (Optional[torch.Tensor], optional):
                マスクテンソル (B, 1, T)

        Returns:
            torch.Tensor: 出力テンソル (B, channels, T)
        """
        # 2つの畳み込み層を順次適用
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            # Residual接続
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """
        Weight Normalizationを削除
        """
        for layer in self.convs:
            remove_weight_norm(layer)


class SameConvBlock(nn.Module):
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
