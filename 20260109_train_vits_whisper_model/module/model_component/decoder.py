from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from utils.model import init_weights
from module.model_component.modules import ResBlock1, ResBlock2, LRELU_SLOPE


class Generator(torch.nn.Module):
    """
    VITS Generator (Decoder) - 音声波形生成用デコーダ

    転置畳み込み（アップサンプリング層）と残差ブロックを組み合わせて、
    音響特徴量から生の音声波形を生成します。話者埋め込みによる
    条件付き生成にも対応しています。

    アーキテクチャ:
        1. 前処理畳み込み: 入力特徴量を初期隠れ次元にマッピング
        2. アップサンプリング層: ConvTranspose1dによる段階的なアップサンプリング
        3. 残差ブロック: 各アップサンプリング後に複数の並列残差ブロック
        4. 後処理畳み込み: 単一チャンネルの波形を生成する最終層
    """

    def __init__(
        self,
        initial_channel: int,
        resblock_str: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int = 0,
    ) -> None:
        """
        Generatorモジュールを初期化します。

        Args:
            initial_channel (int): 入力特徴量のチャンネル数
            resblock_str (str): 残差ブロックのタイプ
                ("1" でResBlock1、"2" でResBlock2)
            resblock_kernel_sizes (list[int]): 各残差ブロックのカーネルサイズ
            resblock_dilation_sizes (list[list[int]]): 各残差ブロックの
                dilation率
            upsample_rates (list[int]): 各転置畳み込み層の
                アップサンプリング率
            upsample_initial_channel (int): アップサンプリング前の
                初期チャンネル数
            upsample_kernel_sizes (list[int]): 各アップサンプリング層の
                カーネルサイズ
            gin_channels (int, optional): グローバル条件付け（話者埋め込み）
                のチャンネル数。0の場合、条件付けは適用されません。
                デフォルト: 0
        """
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # 前処理畳み込み層: 入力特徴量を初期チャンネル数にマッピング
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        # 残差ブロックのタイプを選択
        resblock = ResBlock1 if resblock_str == "1" else ResBlock2

        # アップサンプリング層を構築
        # 各層で徐々にチャンネル数を半減させながら時間解像度を上げる
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # 残差ブロックを構築
        # 各アップサンプリング層の後に複数の残差ブロックを配置
        self.resblocks = nn.ModuleList()
        ch = None
        for i in range(len(self.ups)):
            # 現在のアップサンプリング層の出力チャンネル数を計算
            ch = upsample_initial_channel // (2 ** (i + 1))
            # 各カーネルサイズとdilationサイズの組み合わせで
            # 残差ブロックを追加
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))  # type: ignore

        # 後処理畳み込み層: 最終的に1チャンネルの波形を生成
        assert ch is not None
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

        # アップサンプリング層の重みを初期化
        self.ups.apply(init_weights)

        # グローバル条件付け（話者埋め込み）のための畳み込み層
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def _apply_resblocks(
        self, x: torch.Tensor, upsample_idx: int
    ) -> torch.Tensor:
        """
        複数の残差ブロックを並列に適用し、出力を平均化します。

        Args:
            x (torch.Tensor): アップサンプリング後の入力テンソル
            upsample_idx (int): 現在のアップサンプリング層のインデックス

        Returns:
            torch.Tensor: 全ての残差ブロックの出力を平均化したテンソル
        """
        # 各残差ブロックの出力を集約
        xs = None
        for j in range(self.num_kernels):
            # 残差ブロックのインデックスを計算
            resblock_idx = upsample_idx * self.num_kernels + j
            if xs is None:
                xs = self.resblocks[resblock_idx](x)
            else:
                xs += self.resblocks[resblock_idx](x)

        assert xs is not None
        # 全ての残差ブロックの出力を平均化
        return xs / self.num_kernels

    def forward(
        self, x: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        音響特徴量から波形を生成します。

        Args:
            x (torch.Tensor): 入力音響特徴量
                Shape: (batch, initial_channel, time)
            g (torch.Tensor, optional): グローバル条件付け（話者埋め込み）
                Shape: (batch, gin_channels, time)

        Returns:
            torch.Tensor: 生成された音声波形
                Shape: (batch, 1, time * prod(upsample_rates))
        """
        # 前処理: 入力特徴量を変換
        x = self.conv_pre(x)

        # グローバル条件付け（話者埋め込み）を適用
        if g is not None:
            x = x + self.cond(g)

        # アップサンプリング + 残差ブロックを順次適用
        for i in range(self.num_upsamples):
            # 活性化関数を適用
            x = F.leaky_relu(x, LRELU_SLOPE)
            # アップサンプリング
            x = self.ups[i](x)
            # 複数の残差ブロックを並列に適用して平均化
            x = self._apply_resblocks(x, i)

        # 後処理: 最終的な波形を生成
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)  # [-1, 1] の範囲にクリップ

        return x

    def remove_weight_norm(self) -> None:
        """
        全ての層から重み正規化を除去します。

        訓練後に推論速度を向上させるため、アップサンプリング層と
        残差ブロック層から重み正規化のラッパーを除去します。
        このメソッドは訓練完了後に呼び出してください。
        """
        print("Removing weight norm...")
        # アップサンプリング層から重み正規化を除去
        for layer in self.ups:
            remove_weight_norm(layer)
        # 残差ブロックから重み正規化を除去
        for layer in self.resblocks:
            layer.remove_weight_norm()
