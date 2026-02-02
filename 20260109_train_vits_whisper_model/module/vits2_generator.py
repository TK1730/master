from typing import Optional

import torch
import torch.nn as nn

from module.model_component.text_encoder import TextEncoder
from module.model_component.posterior_encoder import PosteriorEncoder
from module.model_component.normalizing_flows import (
    ResidualCouplingBlock,
    TransformerCouplingBlock,
)
from module.model_component.duration_predictors import (
    StochasticDurationPredictor,
    DurationPredictor,
)
from module.model_component.decoder import Generator
from utils.monotonic_align import search_path, generate_path
from utils.model import sequence_mask, rand_slice_segments


class Vits2Generator(nn.Module):
    """
    VITS2 音声合成モデル

    VITS2は、Variational Inference with adversarial learning for end-to-end
    Text-to-Speech (VITS) の改良版で、以下の特徴を持ちます：

    主要コンポーネント:
        - TextEncoder: 音素列を潜在表現に変換
        - PosteriorEncoder: メルスペクトログラムを潜在表現に変換
        - Generator (Decoder): 潜在変数から音声波形を生成
        - Flow: テキストと音声の潜在空間を相互変換
        - DurationPredictor: 音素の持続時間を予測 (SDP + DP)

    改良点:
        - Transformer-based Flow (オプション)
        - Stochastic Duration Predictor (SDP)
        - Monotonic Alignment Search with noise scaling
        - 話者条件付きエンコーダ
    """

    def __init__(
        self,
        # 共通パラメータ
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        n_speakers: int = 0,
        gin_channels: int = 0,
        # TextEncoder用パラメータ
        text_enc_hidden_channels: int = 192,
        text_enc_filter_channels: int = 768,
        text_enc_n_heads: int = 2,
        text_enc_n_layers: int = 6,
        text_enc_kernel_size: int = 3,
        text_enc_p_dropout: float = 0.1,
        # PosteriorEncoder用パラメータ
        post_enc_n_layers: int = 16,
        post_enc_kernel_size: int = 5,
        post_enc_dilation_rate: int = 1,
        # Decoder (Generator)用パラメータ
        decoder_resblock_type: str = "1",
        decoder_resblock_kernel_sizes: list[int] = None,
        decoder_resblock_dilation_sizes: list[list[int]] = None,
        decoder_upsample_rates: list[int] = None,
        decoder_upsample_initial_channel: int = 512,
        decoder_upsample_kernel_sizes: list[int] = None,
        # Flow用パラメータ
        flow_n_layers: int = 4,
        flow_transformer_n_layers: int = 6,
        flow_share_parameter: bool = False,
        flow_kernel_size: int = 5,
        use_transformer_flow: bool = True,
        # DurationPredictor用パラメータ
        sdp_filter_channels: int = 192,
        sdp_kernel_size: int = 3,
        sdp_p_dropout: float = 0.5,
        sdp_n_flows: int = 4,
        dp_filter_channels: int = 256,
        dp_kernel_size: int = 3,
        dp_p_dropout: float = 0.5,
        # その他の設定:
        use_sdp: bool = True,
        use_spk_conditoned_encoder: bool = True,
        use_noise_scaled_mas: bool = False,
        mas_noise_scale_initial: float = 0.01,
        noise_scale_delta: float = 2e-6,
    ) -> None:
        """VITS2の全体像

        Args:
            共通パラメータ:
                n_vocab (int): 音素の種類数
                spec_channels (int): スペクトログラムのチャンネル数
                segment_size (int): セグメントサイズ(音声の長さ)
                inter_channels (int): 潜在空間のチャンネル数(旧: latent_channels)
                n_speakers (int): スピーカーの数. Defaults to 0.
                gin_channels (int): 話者埋め込みのチャンネル数. Defaults to 0.

            TextEncoder用パラメータ:
                text_enc_hidden_channels (int): 隠れ層のチャンネル数
                text_enc_filter_channels (int): フィルター層のチャンネル数
                text_enc_n_heads (int): Attention機構のヘッド数
                text_enc_n_layers (int): Transformer層の数
                text_enc_kernel_size (int): 畳み込みカーネルサイズ
                text_enc_p_dropout (float): ドロチE�Eアウト率

            PosteriorEncoder用パラメータ:
                post_enc_n_layers (int): WaveNet残差ブロックの層数
                post_enc_kernel_size (int): 畳み込みカーネルサイズ
                post_enc_dilation_rate (int): Dilation率

            Decoder (Generator)用パラメータ:
                decoder_resblock_type (str): 残差ブロックの種類"1" or "2")
                decoder_resblock_kernel_sizes (list[int]): 残差ブロックの
                    カーネルサイズリスト
                decoder_resblock_dilation_sizes (list[list[int]]): 残差
                    ブロックのdilationサイズリスト
                decoder_upsample_rates (list[int]): アップサンプリング率
                decoder_upsample_initial_channel (int): アップサンプリング
                    初期チャンネル数
                decoder_upsample_kernel_sizes (list[int]): アップサンプリング
                    カーネルサイズ

            Flow用パラメータ:
                flow_n_layers (int): フロー層の数
                flow_transformer_n_layers (int): Transformerフロー使用時の層数
                flow_share_parameter (bool): パラメータ共有するかどうか
                flow_kernel_size (int): 畳み込みカーネルサイズ
                use_transformer_flow (bool): TransformerFlowを使用するか
                    (Falseの場合はResidualCouplingBlock)

            DurationPredictor用パラメータ:
                sdp_filter_channels (int): StochasticDPのフィルターチャンネル数
                sdp_kernel_size (int): StochasticDPのカーネルサイズ
                sdp_p_dropout (float): StochasticDPのドロチE�Eアウト率
                sdp_n_flows (int): StochasticDP内部のフロー数
                dp_filter_channels (int): DurationPredictorのフィルター
                    チャンネル数
                dp_kernel_size (int): DurationPredictorのカーネルサイズ
                dp_p_dropout (float): DurationPredictorのドロチE�Eアウト率

            その他の設定:
                use_sdp (bool): Stochastic Duration Predictorを使用するか
                use_spk_conditoned_encoder (bool): 話者条件付きエンコーダを
                    使用するか
                use_noise_scaled_mas (bool): noise_scaled MASを使用するか
                mas_noise_scale_initial (float): noise_scaled MASの初期値
                noise_scale_delta (float): noise_scaled MASの減衰量
        """
        super().__init__()
        # 共通パラメータを保存
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        # モジュール別パラメータを保存
        self.text_enc_n_layers = text_enc_n_layers
        self.post_enc_n_layers = post_enc_n_layers
        self.decoder_upsample_rates = decoder_upsample_rates
        self.flow_n_layers = flow_n_layers
        self.use_sdp = use_sdp
        self.use_transformer_flow = use_transformer_flow

        # MAS (Monotonic Alignment Search) 設定
        self.use_noise_scaled_mas = use_noise_scaled_mas
        self.mas_noise_scale_initial = mas_noise_scale_initial
        self.noise_scale_delta = noise_scale_delta
        self.current_mas_noise_scale = mas_noise_scale_initial

        # 話者条件付きエンコーダ設定
        self.use_spk_conditoned_encoder = use_spk_conditoned_encoder
        if use_spk_conditoned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            # 話者条件付きが無効な場合は0を設定（Noneだとnn.Linearでエラーになる）
            self.enc_gin_channels = 0

        # デフォルト値を設定 (Noneの場合)
        if decoder_resblock_kernel_sizes is None:
            decoder_resblock_kernel_sizes = [3, 7, 11]
        if decoder_resblock_dilation_sizes is None:
            decoder_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if decoder_upsample_rates is None:
            decoder_upsample_rates = [8, 8, 2, 2]
        if decoder_upsample_kernel_sizes is None:
            decoder_upsample_kernel_sizes = [16, 16, 4, 4]

        # テキストエンコーダ: 音素列を潜在表現に変換
        self.enc_p = TextEncoder(
            n_vocab=n_vocab,
            out_channels=inter_channels,
            hidden_channels=text_enc_hidden_channels,
            filter_channels=text_enc_filter_channels,
            n_heads=text_enc_n_heads,
            n_layers=text_enc_n_layers,
            kernel_size=text_enc_kernel_size,
            p_dropout=text_enc_p_dropout,
            gin_channels=self.enc_gin_channels,
        )

        # 音声エンコーダ: 線形スペクトログラムと話者埋め込みをencode
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=text_enc_hidden_channels,
            kernel_size=post_enc_kernel_size,
            dilation_rate=post_enc_dilation_rate,
            n_layers=post_enc_n_layers,
            gin_channels=gin_channels,
        )

        # デコーダ: z, speaker_id_embeddingを入力に受け取り音声を生成
        self.dec = Generator(
            initial_channel=inter_channels,
            resblock_str=decoder_resblock_type,
            resblock_kernel_sizes=decoder_resblock_kernel_sizes,
            resblock_dilation_sizes=decoder_resblock_dilation_sizes,
            upsample_rates=decoder_upsample_rates,
            upsample_initial_channel=decoder_upsample_initial_channel,
            upsample_kernel_sizes=decoder_upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        # Flow: zと埋め込み済み話老Edを�E力にとり、Monotonic Alignment Search
        # で使用する変数z_pを出力するネットワーク (逆方向も可)
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                channels=inter_channels,
                hidden_channels=text_enc_hidden_channels,
                filter_channels=text_enc_filter_channels,
                n_heads=text_enc_n_heads,
                n_layers=flow_transformer_n_layers,
                kernel_size=flow_kernel_size,
                p_dropout=text_enc_p_dropout,
                n_flows=flow_n_layers,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                channels=inter_channels,
                hidden_channels=text_enc_hidden_channels,
                kernel_size=flow_kernel_size,
                dilation_rate=1,
                n_layers=flow_n_layers,
                gin_channels=gin_channels,
            )

        # 持続時間予測器
        # Stochastic Duration Predictor
        self.sdp = StochasticDurationPredictor(
            in_channels=text_enc_hidden_channels,
            filter_channels=sdp_filter_channels,
            kernel_size=sdp_kernel_size,
            p_dropout=sdp_p_dropout,
            n_flows=sdp_n_flows,
            gin_channels=gin_channels,
        )
        # Deterministic Duration Predictor
        self.dp = DurationPredictor(
            in_channels=text_enc_hidden_channels,
            filter_channels=dp_filter_channels,
            kernel_size=dp_kernel_size,
            p_dropout=dp_p_dropout,
            gin_channels=gin_channels,
        )

        # 話者埋め込み
        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        """
        順伝播（学習時）

        音素列とメルスペクトログラムから音声波形を生成し、学習に必要な
        各種損失を計算します。

        Args:
            x (torch.Tensor): 音素インデックス列
                Shape: [batch, text_len]
            x_lengths (torch.Tensor): 各系列の有効な音素数
                Shape: [batch]
            y (torch.Tensor): メルスペクトログラムまたは線形スペクトログラム
                Shape: [batch, spec_channels, spec_len]
            y_lengths (torch.Tensor): 各スペクトログラムの有効なフレーム数
                Shape: [batch]
            sid (torch.Tensor): 話者ID
                Shape: [batch]

        Returns:
            tuple: 以下の要素を含むタプル
                - wav_fake (torch.Tensor): 生成された音声波形
                    Shape: [batch, 1, segment_size]
                - l_length (torch.Tensor): 持続時間予測の損失
                    Shape: [batch]
                - attn (torch.Tensor): アライメント行列(MASの結果)
                    Shape: [batch, 1, text_len, spec_len]
                - ids_slice (torch.Tensor): スライスされたセグメントの開始位置
                    Shape: [batch]
                - x_mask (torch.Tensor): テキストのマスク
                    Shape: [batch, 1, text_len]
                - y_mask (torch.Tensor): スペクトログラムのマスク
                    Shape: [batch, 1, spec_len]
                - (z, z_p, m_p, logs_p, m_q, logs_q): 潜在変数のタプル
                    - z: PosteriorEncoderからの潜在変数
                    - z_p: Flowで変換後の潜在変数
                    - m_p: テキストからの平均（アライメント後）
                    - logs_p: テキストからの対数分散（アライメント後）
                    - m_q: スペクトログラムからの平均
                    - logs_q: スペクトログラムからの対数分散
                - (x, logw, logw_): 持続時間関連のタプル
                    - x: エンコードされたテキスト特徴量
                    - logw: 予測された対数持続時間
                    - logw_: 正解の対数持続時間(MASから)
                - g: 話者埋め込みベクトル
        """
        # 話者埋め込みベクトルを取得
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, gin_channels, 1]
        else:
            g = None

        # テキストエンコーダ: 音素列を潜在表現に変換
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        # 音声エンコーダ: スペクトログラムを潜在表現に変換
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # Flow: 音声潜在変数をテキスト潜在空間に変換（逆方向）
        z_p = self.flow(z, y_mask, g=g)

        # Monotonic Alignment Search (MAS)
        # テキストと音声の最適なアライメントを探索
        attn = search_path(
            z_p,
            m_p,
            logs_p,
            x_mask,
            y_mask,
            mas_noise_scale=self.current_mas_noise_scale,
            use_mas_noise_scale=self.use_noise_scaled_mas,
        )

        # 正解の持続時間（アライメントベース）
        # アライメント行列から各音素の持続時間を計算
        w = attn.sum(2)  # [b, 1, text_len]

        # Stochastic Duration Predictor (SDP) の損失
        # VAE形式で持続時間を予測
        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        # 正解の対数持続時間
        logw_ = torch.log(w + 1e-6) * x_mask

        # Deterministic Duration Predictor (DP) による予測
        logw = self.dp(x, x_mask, g=g)

        # 平均二乗誤差を計算
        l_length_dp = torch.sum(
            (logw - logw_) ** 2, [1, 2]
        ) / torch.sum(x_mask)

        # 持続時間に関する合計損失（SDPとDPの和）
        l_length = l_length_dp + l_length_sdp

        # テキストの潜在変数を持続時間に基づいて拡張
        # アライメント行列を使って音素レベルからフレームレベルに変換
        m_p = torch.matmul(attn.squeeze(1), m_p.mT).mT
        logs_p = torch.matmul(attn.squeeze(1), logs_p.mT).mT

        # ランダムにセグメントをスライス（学習効率化のため）
        z_slice, ids_slice = rand_slice_segments(
            z, y_lengths, self.segment_size
        )

        # デコーダで音声波形を生成
        wav_fake = self.dec(z_slice, g=g)

        return (
            wav_fake,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
            g,
        )

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor = None,
        noise_scale: float = 0.667,
        length_scale: float = 1,
        noise_scale_w: float = 0.8,
        max_len: Optional[int] = None,
        sdp_ratio: float = 0.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        """
        推論（音声合成）

        テキストから音声を生成します。持続時間はモデルが予測します。

        Args:
            x (torch.Tensor): 音素インデックス列
                Shape: [batch, text_len]
            x_lengths (torch.Tensor): 各系列の有効な音素数
                Shape: [batch]
            sid (torch.Tensor, optional): 話者ID
                Shape: [batch]. Defaults to None.
            noise_scale (float): Flow逆変換時のnoize_scale
                大きいほど多様性が増す。Defaults to 0.667.
            length_scale (float): 持続時間のスケール
                1.0より大きいと遅く、小さいと速くなる。Defaults to 1.
            noise_scale_w (float): 持続時間予測時のノイズスケール
                Defaults to 0.8.
            max_len (Optional[int]): 生成する最大長（フレーム数）
                Noneの場合は制限なし。Defaults to None.
            sdp_ratio (float): SDPとDPの混合比率
                0.0でDP、1.0でSDPのみ使用。Defaults to 0.0.

        Returns:
            tuple: 以下の要素を含むタプル
                - o (torch.Tensor): 生成された音声波形
                    Shape: [batch, 1, audio_len]
                - attn (torch.Tensor): アライメント行列
                    Shape: [batch, 1, text_len, spec_len]
                - y_mask (torch.Tensor): スペクトログラムのマスク
                    Shape: [batch, 1, spec_len]
                - (z_p_dur, m_p_dur, logs_p_dur): 潜在変数のタプル
        """
        # 話者埋め込みベクトルを取得
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, gin_channels, 1]
        else:
            g = None

        # テキストエンコーダ: 音素列を潜在表現に変換
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        # 持続時間予測
        # SDPとDPを混合して持続時間を予測
        logw = self.sdp(
            x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w
        ) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)

        # 持続時間からフレーム数を計算
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)  # 整数にする
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            sequence_mask(y_lengths, None), 1
        ).to(x_mask.dtype)

        # アライメント行列を生成（ハードアテンション）
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        # テキストの潜在変数を持続時間に基づいて拡張
        m_p_dur = torch.matmul(
            attn.squeeze(1), m_p.mT
        ).mT  # [b, text_len, spec_len] @ [b, text_len, latent]
        #       -> [b, latent, spec_len]
        logs_p_dur = torch.matmul(
            attn.squeeze(1), logs_p.mT
        ).mT  # [b, text_len, spec_len] @ [b, text_len, latent]
        #       -> [b, latent, spec_len]

        # テキストの潜在変数をサンプリング
        z_p_dur = (
            m_p_dur + torch.randn_like(m_p_dur) * torch.exp(logs_p_dur)
            * noise_scale
        )

        # Flow: テキスト潜在空間から音声潜在空間に変換（順方向）
        z = self.flow(z_p_dur, y_mask, g=g, reverse=True)

        # デコーダで音声波形を生成
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return (
            o,
            attn,
            y_mask,
            (z_p_dur, m_p_dur, logs_p_dur)
        )

    @torch.no_grad()
    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        """
        声質変換

        ソース話者の音声を、ターゲット話者の声質に変換します。

        Args:
            y (torch.Tensor): 入力音声のスペクトログラム
                Shape: [batch, spec_channels, spec_len]
            y_lengths (torch.Tensor): 各スペクトログラムの有効なフレーム数
                Shape: [batch]
            sid_src (torch.Tensor): ソース話者のID
                Shape: [batch]
            sid_tgt (torch.Tensor): ターゲット話者のID
                Shape: [batch]

        Returns:
            tuple: 以下の要素を含むタプル
                - o_hat (torch.Tensor): 変換後の音声波形
                    Shape: [batch, 1, audio_len]
                - y_mask (torch.Tensor): スペクトログラムのマスク
                    Shape: [batch, 1, spec_len]
                - z_q_dur: ソース話者の潜在変数
                - z_p_dur: ターゲット話者の潜在変数
        """
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."

        # ソースとターゲットの話者埋め込みを取得
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)

        # 音声エンコーダ: ソース音声を潜在表現に変換
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.enc_q(
            y, y_lengths, g=g_src
        )

        # Flow: 音声潜在変数をテキスト潜在空間に変換（ソース話者）
        z_q_dur = self.flow(
            z_q_audio, y_mask, g=g_src
        )

        # Flow: テキスト潜在空間から音声潜在空間に変換（ターゲット話者）
        z_p_dur = self.flow(
            z_q_dur, y_mask, g=g_tgt, reverse=True
        )

        # デコーダ: ターゲット話者の声質で音声を生成
        o_hat = self.dec(z_p_dur * y_mask, g=g_tgt)
        return (
            o_hat,
            y_mask,
            z_q_dur,
            z_p_dur,
        )


if __name__ == "__main__":
    # 動作確誁E
    model = Vits2Generator(
        n_vocab=10,
        spec_channels=80,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        n_layers_q=3,
        n_flows=4,
        kernel_size=3,
        p_dropout=0.1,
        speaker_cond_layer=0,
        resblock="1",
        resblock_kernel_sizes=[3, 5, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        mas_noise_scale=1.0,
        noise_scale_delta=2e-6,
        use_transformer_flow=True,
        n_speakers=10,
        gin_channels=192,
    )
    x = torch.randint(0, 10, (2, 50))
    x_lengths = torch.tensor([50, 45])
    y = torch.randn(2, 80, 200)
    y_lengths = torch.tensor([200, 180])
    sid = torch.tensor([3, 5])
    (
        wav_fake,
        l_length,
        attn,
        ids_slice,
        x_mask,
        y_mask,
        (z, z_p, m_p, logs_p, m_q, logs_q),
        (x, logw, logw_),
        g,
    ) = model(x, x_lengths, y, y_lengths, sid)
    print(wav_fake.shape)
