import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

from module.model_component.transformer import RelativePositionTransformer
from utils.model import sequence_mask


class TextEncoder(nn.Module):
    """
    VITSモデルのテキストエンコーダ

    音素系列を入力として受け取り、音響特徴の事前分布パラメータ(平均と分散)
    を出力します。内部でRelativePositionTransformerを使用して、
    テキストの文脈情報を捉えます。

    処理の流れ:
        1. 音素インデックスを埋め込みベクトルに変換
        2. Transformer層で文脈情報を抽出
        3. 線形射影により平均(m)と対数分散(logs)を予測

    多話者モデルの場合、gin_channelsを0以外に設定することで
    話者条件付けが有効になります。
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        """
        テキストエンコーダの初期化

        Args:
            n_vocab (int): 音素の種類数(語彙サイズ)
            out_channels (int): 出力のチャネル数。VAEの潜在変数次元に対応
            hidden_channels (int): Transformer内部の隠れ層チャネル数
            filter_channels (int): Transformer FFN層の中間チャネル数
            n_heads (int): マルチヘッドアテンションのヘッド数
            n_layers (int): Transformer層の積層数
            kernel_size (int): FFN層の畳み込みカーネルサイズ
            p_dropout (float): ドロップアウト率(0.0～1.0)
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                0の場合は単一話者モデル、0以外の場合は多話者モデルとして動作。
                デフォルトは0。
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_head = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 音素埋め込み層
        # 音素インデックスをhidden_channels次元のベクトルに変換
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        # Xavier初期化の変形(標準偏差 = hidden_channels^-0.5)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # Transformerエンコーダ
        # 相対位置エンコーディングを使用して文脈情報を抽出
        self.encoder = RelativePositionTransformer(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        # 射影層: Transformer出力を平均と対数分散に変換
        # out_channels * 2 は、平均(m)とlog分散(logs)のため
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        テキストエンコーダの順伝播

        音素系列から音響特徴の事前分布パラメータを計算します。

        Args:
            x (torch.Tensor): 音素インデックス系列 (B, T)
                - B: バッチサイズ
                - T: 系列長
            x_lengths (torch.Tensor): 各バッチの有効系列長 (B,)
                パディングを除いた実際の系列長を示す
            g (Optional[torch.Tensor], optional): 話者埋め込みベクトル
                (B, gin_channels)。gin_channels=0の場合は不要。
                デフォルトはNone。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - x: Transformer出力 (B, hidden_channels, T)
                - m: 事前分布の平均 (B, out_channels, T)
                - logs: 事前分布の対数分散 (B, out_channels, T)
                - x_mask: パディングマスク (B, 1, T)
        """
        # 1. 音素埋め込み + スケーリング
        # スケーリング係数 sqrt(hidden_channels) はTransformerの標準的な手法
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]

        # 2. 次元の転置: (B, T, H) -> (B, H, T)
        # Conv1dはチャネル次元が2番目である必要があるため
        x = torch.transpose(x, 1, -1)  # [B, H, T]

        # 3. パディングマスクの生成
        # x_lengthsに基づいて有効な位置を1、パディング位置を0にする
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)  # [B, 1, T]

        # 4. Transformerエンコーダで文脈情報を抽出
        # マスクを適用して、パディング部分が処理に影響しないようにする
        x = self.encoder(x * x_mask, x_mask, g=g)  # [B, H, T]

        # 5. 線形射影で事前分布のパラメータを予測
        stats = self.proj(x) * x_mask  # [B, out_channels*2, T]

        # 6. 平均と対数分散に分割
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # m: [B, out_channels, T], logs: [B, out_channels, T]

        return x, m, logs, x_mask
