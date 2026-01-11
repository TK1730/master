from typing import Tuple
import torch
import torch.nn as nn

from module.model_component.transformer import RelativePositionTransformer
from utils.model import sequence_mask

"""
話者特徴量を抽出するエンコーダー
FreeVC, so-vits-svcなどで使用される

未知の話者(zero-shot)に対応するために拡張されたVITS派生モデル
実験では未使用
"""


class SpeakerEncoder(torch.nn.Module):
    """
    LSTM ベースの話者エンコーダ

    メルスペクトログラムから話者埋め込みベクトルを抽出します。
    長い音声に対しては、複数のセグメントから埋め込みを生成して平均化することで、
    より安定した話者表現を得ることができます。

    主な用途:
        - Zero-shot音声変換(FreeVC, so-vits-svcなど)
        - 未知の話者に対する話者埋め込みの生成

    注意:
        本実験では未使用のモジュールです。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
    ) -> None:
        """
        SpeakerEncoderの初期化

        Args:
            in_channels (int): 入力メルスペクトログラムのチャネル数
                （通常はメルフィルタバンクの数）
            out_channels (int): 出力話者埋め込みベクトルの次元数
            hidden_channels (int): LSTMの隠れ層のチャネル数
            n_layers (int): LSTMの積層数
        """
        super().__init__()
        # 時系列特徴抽出用のLSTM層
        self.lstm = nn.LSTM(
            in_channels,
            hidden_channels,
            n_layers,
            batch_first=True  # (B, T, C)の形式で入力を受け取る
        )
        # 話者埋め込み次元への射影層
        self.linear = nn.Linear(hidden_channels, out_channels)
        # 非線形活性化関数
        self.relu = nn.ReLU()

    def forward(
        self,
        mels: torch.Tensor,
    ) -> torch.Tensor:
        """
        メルスペクトログラムから正規化された話者埋め込みを生成

        Args:
            mels (torch.Tensor): メルスペクトログラム (B, T, C)
                - B: バッチサイズ
                - T: 時間フレーム数
                - C: メルフィルタバンクの数

        Returns:
            torch.Tensor: L2正規化された話者埋め込みベクトル (B, out_channels)
        """
        # LSTM のパラメータを連続したメモリに配置（高速化のため）
        self.lstm.flatten_parameters()
        # LSTM で時系列をエンコード、最終層の隠れ状態を取得
        _, (hidden, _) = self.lstm(mels)
        # 最終層の隠れ状態を線形変換 + ReLU活性化
        embeds_raw = self.relu(self.linear(hidden[-1]))
        # L2正規化：埋め込みベクトルを単位球面上に射影
        # これにより話者間の距離をコサイン類似度で測定可能にする
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(
        self,
        total_frames: int,
        partial_frames: int,
        partial_hop: int,
    ) -> list:
        """
        長い音声を部分セグメントに分割するためのインデックスリストを生成

        スライディングウィンドウ方式で音声を分割し、各セグメントの
        フレームインデックスを返します。

        Args:
            total_frames (int): 入力音声の総フレーム数
            partial_frames (int): 各セグメントのフレーム数
            partial_hop (int): セグメント間のホップサイズ（ストライド）

        Returns:
            list: 各セグメントのフレームインデックスのリスト
        """
        mel_slices = []
        # スライディングウィンドウでセグメントのインデックスを生成
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(
        self,
        mel: torch.Tensor,
        partial_frames: int = 128,
        partial_hop: int = 64,
    ) -> torch.Tensor:
        """
        発話全体から安定した話者埋め込みを生成

        長い音声に対しては、複数のセグメントから埋め込みを抽出して平均化し、
        より頑健な話者表現を得ます。短い音声の場合は末尾のセグメントのみを使用します。

        Args:
            mel (torch.Tensor): メルスペクトログラム (1, T, C)
            partial_frames (int, optional): 各セグメントのフレーム数。
                デフォルトは128。
            partial_hop (int, optional): セグメント間のホップサイズ。
                デフォルトは64。

        Returns:
            torch.Tensor: 話者埋め込みベクトル (1, out_channels)
        """
        mel_len = mel.size(1)
        # 末尾のセグメントを取得（短い音声用のフォールバック）
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            # 長い音声の場合：複数セグメントに分割
            mel_slices = self.compute_partial_slices(
                mel_len, partial_frames, partial_hop
            )
            mels = list(mel[:, s] for s in mel_slices)
            # 末尾のセグメントも追加（完全なカバレッジのため）
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            # 各セグメントから埋め込みを抽出（勾配計算なし）
            with torch.no_grad():
                partial_embeds = self(mels)
            # 全セグメントの埋め込みを平均化
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # 注: 既にforwardで正規化済みのため、再正規化は不要
        else:
            # 短い音声の場合：末尾セグメントのみ使用
            with torch.no_grad():
                embed = self(last_mel)

        return embed


class AudioEncoder(torch.nn.Module):
    """
    Transformer ベースのオーディオエンコーダ

    音響特徴量を潜在表現にエンコードし、変分推論のための
    平均と分散パラメータを出力します。相対位置エンコーディングを使用した
    Transformerにより、長距離依存関係を効果的にモデル化します。

    主な特徴:
        - 相対位置エンコーディングによる位置情報の学習
        - 話者埋め込みによる条件付け（多話者モデル対応）
        - 変分オートエンコーダ(VAE)の事後分布パラメータを出力

    アーキテクチャ:
        入力 -> Pre-Conv -> Transformer Encoder -> Post-Conv -> (平均, log分散)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_fft: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dorpout: float,
        gin_channels: int,
        speaker_cond_layer: int = 0,
    ) -> None:
        """
        AudioEncoderの初期化

        Args:
            in_channels (int): 入力特徴量のチャネル数
            out_channels (int): 出力潜在表現のチャネル数
            hidden_channels (int): Transformer層の隠れ層チャネル数
            hidden_channels_fft (int): Feed-Forward Network(FFN)の
                中間層チャネル数
            n_heads (int): マルチヘッドアテンションのヘッド数
            n_layers (int): Transformer層の積層数
            kernel_size (int): FFN層の畳み込みカーネルサイズ
            p_dorpout (float): ドロップアウト率
            gin_channels (int): 話者埋め込みのチャネル数
                (0の場合は単一話者モデル)
            speaker_cond_layer (int, optional): 話者埋め込みを加算する層の
                インデックス。デフォルトは0。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_channels_fft = hidden_channels_fft
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dorpout = p_dorpout
        self.gin_channels = gin_channels
        self.speaker_cond_layer = speaker_cond_layer

        # 入力特徴量を隠れ層の次元に射影
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # 相対位置Transformerエンコーダ
        self.encoder = RelativePositionTransformer(
            hidden_channels=hidden_channels,
            filter_channels=hidden_channels_fft,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dorpout=p_dorpout,
            gin_channels=gin_channels,
        )

        # 平均と対数分散を出力する射影層（*2はmとlogsの2つ分）
        self.post = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        音響特徴量を潜在表現にエンコード

        変分オートエンコーダ(VAE)の事後分布 q(z|x) をモデル化します。
        出力された平均と分散パラメータから、再パラメータ化トリックにより
        潜在変数zをサンプリングします。

        Args:
            x (torch.Tensor): 入力特徴量 (B, in_channels, T)
                - B: バッチサイズ
                - in_channels: 入力チャネル数
                - T: 時系列長
            x_lengths (torch.Tensor): 各サンプルの有効長 (B,)
            g (torch.Tensor, optional): 話者埋め込み (B, gin_channels)。
                多話者モデルの場合に使用。デフォルトはNone。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - z (torch.Tensor): サンプリングされた潜在変数 (B, out_channels, T)
                - m (torch.Tensor): 事後分布の平均 (B, out_channels, T)
                - logs (torch.Tensor): 事後分布の対数標準偏差 (B, out_channels, T)
                - x_mask (torch.Tensor): パディングマスク (B, 1, T)
        """
        # 有効長からパディングマスクを生成 (B, 1, T)
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        # 入力を隠れ層次元に射影し、マスク適用
        x = self.pre(x) * x_mask
        # Transformerエンコーダで特徴抽出（話者埋め込みで条件付け）
        x = self.encoder(x, x_mask, g=g)
        # 平均と対数分散を出力（stats: [B, out_channels*2, T]）
        stats = self.post(x) * x_mask
        # 平均(m)と対数標準偏差(logs)に分割
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # 再パラメータ化トリック: z = μ + σ * ε （ε ~ N(0,1)）
        # これにより勾配を平均と分散パラメータに逆伝播可能
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask
