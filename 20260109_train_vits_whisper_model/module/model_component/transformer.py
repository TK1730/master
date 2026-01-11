import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from utils.model import (
    convert_pad_shape,
    fused_add_tanh_sigmoid_multiply,
    subsequent_mask,
)
from module.model_component.normalization import LayerNorm


class RelativePositionTransformer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        gin_channels: int = 0,
        cond_layer_idx: Optional[int] = None,
    ) -> None:
        """
        相対位置エンコーディングを使用するTransformerエンコーダ

        絶対位置ではなく、トークン間の相対的な距離に基づく
        位置情報を学習するTransformerブロックです。
        複数のマルチヘッドアテンション層とFFN層を積層し、
        オプションで話者埋め込みによる条件付けをサポートします。

        Args:
            hidden_channels (int): 埋め込みとアテンション層のチャネル数
            filter_channels (int): FFN層の中間層のチャネル数
            n_heads (int): マルチヘッドアテンションのヘッド数
            n_layers (int): Transformer層(Attention + FFN)の積層数
            kernel_size (int, optional): FFN層の畳み込みカーネルサイズ
                (default: 1)
            p_dropout (float, optional): ドロップアウト率 (default: 0.0)
            window_size (int, optional): 相対位置エンコーディングの
                ウィンドウサイズ。中心から前後window_size分の範囲を
                学習します (default: 4)
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                0の場合は条件付けなし (default: 0)
            cond_layer_idx (Optional[int], optional): 話者埋め込みを
                加算する層のインデックス。Noneの場合はVITS2の
                デフォルト値2を使用 (default: None)
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.gin_channels = gin_channels

        # vits2では3層目に条件付けを行うためデフォルトで2に設定
        if self.gin_channels != 0:
            self.cond_layer_idx = (
                cond_layer_idx if cond_layer_idx is not None else 2
            )
            self.spk_emb_linear = nn.Linear(
                self.gin_channels,
                hidden_channels
            )
            assert (
                self.cond_layer_idx < self.n_layers
            ), "cond_layer_idx must be less than n_layers"
        else:
            self.cond_layer_idx = self.n_layers

        # Modules
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        # transformer layers
        for i in range(self.n_layers):
            # attention
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            # layer norm
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            # feed forward
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            # layer norm
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Transformerエンコーダの順伝播

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)
            g (torch.Tensor, optional): 話者埋め込みテンソル
                (B, gin_channels)
        Returns:
            torch.Tensor: 出力テンソル (B, C, T)
        """
        # Self-attention用のマスクを作成 (B, 1, T, T)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        # 各Transformer層を順次適用
        for i in range(self.n_layers):
            # 指定された層で話者埋め込みを加算
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                assert g is not None
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask

            # Multi-Head Attention
            y = self.attn_layers[i](x, x, attn_mask=attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)  # Residual接続 + LayerNorm

            # Feed-Forward Network
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)  # Residual接続 + LayerNorm

        x = x * x_mask
        return x


class FFT(nn.Module):
    """
    Feed-Forward Transformer (FFT)

    Self-AttentionとFeed-Forward Networkを積層したTransformerアーキテクチャです。
    正規化フロー(Normalizing Flow)において、話者条件付けを行うための
    特別な機構（isflowモード）をサポートしています。

    主な特徴:
        - Proximal bias/initのサポート（近接位置への注意を促進）
        - 因果的マスクを使用した自己回帰的な生成
        - 正規化フロー用の話者条件付け(isflow=True時)

    isflowモード:
        isflow=Trueの場合、各層に話者情報をFiLM(Feature-wise Linear Modulation)
        によって条件付けします。これにより、多話者モデルで話者固有の
        特徴を学習できます。
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int = 1,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        isflow: bool = False,
        gin_channels: int = 0,
    ) -> None:
        """
        FFT (Feed-Forward Transformer)の初期化

        Args:
            hidden_channels (int): 隠れ層のチャネル数
            filter_channels (int): FFN層の中間チャネル数
            n_heads (int): マルチヘッドアテンションのヘッド数
            n_layers (int, optional): Transformer層の積層数。デフォルトは1。
            kernel_size (int, optional): FFN層の畳み込みカーネルサイズ。
                デフォルトは1。
            p_dropout (float, optional): ドロップアウト率。デフォルトは0.0。
            proximal_bias (bool, optional): 近接位置への注意を促すバイアスを
                使用するか。デフォルトはFalse。
            proximal_init (bool, optional): KeyとQueryの重みを同じ値で
                初期化するか。デフォルトはTrue。
            isflow (bool, optional): 正規化フロー用の話者条件付けを
                有効にするか。Trueの場合、gin_channelsが必要。
                デフォルトはFalse。
            gin_channels (int, optional): 話者埋め込みのチャネル数。
                isflow=Trueの場合に使用。0の場合は単一話者、0以外の場合は
                多話者モデル。デフォルトは0。
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        # 正規化フロー用の話者条件付け層の設定
        if isflow:
            # 話者埋め込みを各層への条件付けパラメータに変換
            # 2*hidden_channels*n_layers: 各層に対してscaleとshiftパラメータ
            cond_layer = torch.nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                1,
            )
            # 入力を条件付け用に2倍のチャネルに変換
            self.cond_pre = torch.nn.Conv1d(
                hidden_channels, 2 * hidden_channels, 1
            )
            # Weight Normalizationを適用して学習を安定化
            self.cond_layer = weight_norm(cond_layer, name="weight")
            self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)

        # 各層のモジュールリストを初期化
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()

        # n_layers分のTransformer層を積層
        for i in range(self.n_layers):
            # Self-Attention層
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            # Attention後のLayer Normalization
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            # Feed-Forward Network (因果的パディングを使用)
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            # FFN後のLayer Normalization
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        FFTの順伝播

        因果的マスクを使用した自己回帰的なTransformer処理を行います。
        isflow=Trueで初期化された場合、話者埋め込みgを使って
        各層でFiLM条件付けを適用します。

        Args:
            x (torch.Tensor): 入力テンソル (B, hidden_channels, T)
                - B: バッチサイズ
                - hidden_channels: チャネル数
                - T: 時系列長
            x_mask (torch.Tensor): パディングマスク (B, 1, T)
                有効な位置を1、パディング位置を0とするマスク
            g (Optional[torch.Tensor], optional): 話者埋め込みベクトル
                (B, gin_channels)。isflow=Trueの場合に使用。
                デフォルトはNone。

        Returns:
            torch.Tensor: 変換後のテンソル (B, hidden_channels, T)
        """
        # 話者条件付けパラメータの準備(isflow=True時のみ)
        if g is not None:
            # 話者埋め込みを各層への条件付けパラメータに変換
            g = self.cond_layer(g)  # [B, 2*hidden_channels*n_layers, T]

        # 因果的マスクの生成(未来の情報を見ないように)
        # subsequent_mask: 下三角行列で、位置iは位置0～iまでのみ参照可能
        self_attn_mask = subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        x = x * x_mask

        # 各Transformer層を順次適用
        for i in range(self.n_layers):
            # 話者条件付け(FiLM: Feature-wise Linear Modulation)
            if g is not None:
                # 入力を2倍のチャネルに拡張
                x = self.cond_pre(x)  # [B, 2*hidden_channels, T]
                # 現在の層用の条件付けパラメータを抽出
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[
                    :, cond_offset: cond_offset + 2 * self.hidden_channels, :
                ]
                # FiLM変換: x = tanh(scale1 * x) * sigmoid(scale2 * x)
                # scale1とscale2はg_lから計算される
                x = fused_add_tanh_sigmoid_multiply(
                    x, g_l, torch.IntTensor([self.hidden_channels])
                )

            # Self-Attention
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)  # Residual接続 + LayerNorm

            # Feed-Forward Network
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)  # Residual接続 + LayerNorm

        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        """
        相対位置エンコーディング対応のマルチヘッドアテンション層

        標準的なマルチヘッドアテンションに加えて、以下の拡張機能を
        サポートします:
        - 相対位置エンコーディング(window_size指定時)
        - 近接位置への注意を促すProximal bias
        - ローカルアテンション(block_length指定時)

        Args:
            channels (int): 入力埋め込みのチャネル数
                (n_headsで割り切れる必要があります)
            out_channels (int): 出力チャネル数
            n_heads (int): マルチヘッドアテンションのヘッド数
            p_dropout (float, optional): ドロップアウト率 (default: 0.0)
            window_size (Optional[int], optional): 相対位置エンコーディングの
                ウィンドウサイズ。Noneの場合は使用しない (default: None)
            heads_share (bool, optional): 相対位置埋め込みをヘッド間で共有
                するか。Trueの場合はパラメータ数削減 (default: True)
            block_length (Optional[int], optional): ローカルアテンションの
                ブロック長。指定時は前後block_length分のみに注意を限定
                (default: None)
            proximal_bias (bool, optional): 近接位置への注意を促すバイアスを
                使用するか (default: False)
            proximal_init (bool, optional): KeyとQueryの重みを同じ値で
                初期化するか (default: False)
        """
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        # attention modules
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        # 相対位置エンコーディング用のパラメータ初期化
        if window_size is not None:
            # ヘッド共有の場合は1、そうでない場合は全ヘッド分のパラメータを用意
            n_heads_rel = 1 if heads_share else n_heads
            # Xavier初期化に基づく標準偏差を計算 (k_channels^-0.5)
            rel_stddev = self.k_channels**-0.5

            # Key用の相対位置埋め込み (形状: [n_heads_rel, 2*window_size+1, k_channels])
            # window_size * 2 + 1 は、中心から前後window_size分の範囲を表現
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            # Value用の相対位置埋め込み (形状: [n_heads_rel, 2*window_size+1, k_channels])
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        # Xavier初期化
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

        # Proximal初期化
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                assert self.conv_k.bias is not None
                assert self.conv_q.bias is not None
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            c (torch.Tensor): コンテキストテンソル (B, C, T)
            attn_mask (Optional[torch.Tensor], optional): アテンションマスクテンソル
        Returns:
            torch.Tensor: 出力テンソル (B, C, T)
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def _reshape_for_attention(
        self, x: torch.Tensor, batch_size: int, seq_length: int
    ) -> torch.Tensor:
        """
        アテンション計算用にテンソルを複数ヘッドに分割し、次元を転置する

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)
            batch_size (int): バッチサイズ
            seq_length (int): シーケンス長

        Returns:
            torch.Tensor: 形状変換後のテンソル (B, n_heads, T, k_channels)
        """
        return x.view(
            batch_size, self.n_heads, self.k_channels, seq_length
        ).transpose(2, 3)

    def _add_relative_position_bias(
        self,
        scores: torch.Tensor,
        query: torch.Tensor,
        t_s: int,
        t_t: int,
    ) -> torch.Tensor:
        """
        相対位置エンコーディングによるバイアスを
        アテンションスコアに加算

        Args:
            scores (torch.Tensor): アテンションスコア
            query (torch.Tensor): クエリテンソル
            t_s (int): キー/バリューのシーケンス長
            t_t (int): クエリのシーケンス長

        Returns:
            torch.Tensor: バイアス加算後のアテンションスコア
        """
        assert t_s == t_t, (
            "Relative attention is only available for self-attention."
        )
        key_relative_embeddings = self._get_relative_embeddings(
            self.emb_rel_k, t_s
        )
        rel_logits = self._matmul_with_relative_keys(
            query / math.sqrt(self.k_channels), key_relative_embeddings
        )
        scores_local = self.s_relative_position_to_absolute_position(
            rel_logits
        )
        return scores + scores_local

    def _add_proximal_bias(
        self, scores: torch.Tensor, t_s: int, t_t: int
    ) -> torch.Tensor:
        """
        近接位置への注意を促すバイアスを
        アテンションスコアに加算

        Args:
            scores (torch.Tensor): アテンションスコア
            t_s (int): キー/バリューのシーケンス長
            t_t (int): クエリのシーケンス長

        Returns:
            torch.Tensor: バイアス加算後のアテンションスコア
        """
        assert t_s == t_t, (
            "Proximal bias is only available for self-attention."
        )
        return scores + self._attention_bias_proximal(t_s).to(
            device=scores.device, dtype=scores.dtype
        )

    def _apply_attention_mask(
        self, scores: torch.Tensor, mask: torch.Tensor, t_s: int, t_t: int
    ) -> torch.Tensor:
        """
        アテンションマスクとブロックマスクを適用

        Args:
            scores (torch.Tensor): アテンションスコア
            mask (torch.Tensor): アテンションマスク
            t_s (int): キー/バリューのシーケンス長
            t_t (int): クエリのシーケンス長

        Returns:
            torch.Tensor: マスク適用後のアテンションスコア
        """
        scores = scores.masked_fill(mask == 0, -1e4)
        if self.block_length is not None:
            assert (
                t_s == t_t
            ), "Local attention is only available for self-attention."
            block_mask = (
                torch.ones_like(scores)
                .triu(-self.block_length)
                .tril(self.block_length)
            )
            scores = scores.masked_fill(block_mask == 0, -1e4)
        return scores

    def _add_relative_position_to_output(
        self, output: torch.Tensor, p_attn: torch.Tensor, t_s: int
    ) -> torch.Tensor:
        """
        Value側の相対位置エンコーディングを出力に追加

        Args:
            output (torch.Tensor): アテンション適用後の出力
            p_attn (torch.Tensor): アテンション確率分布
            t_s (int): キー/バリューのシーケンス長

        Returns:
            torch.Tensor: 相対位置バイアス追加後の出力
        """
        relative_weights = self._absolute_position_to_relative_position(
            p_attn
        )
        value_relative_embeddings = self._get_relative_embeddings(
            self.emb_rel_v, t_s
        )
        return output + self._matmul_with_relative_values(
            relative_weights, value_relative_embeddings
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        マルチヘッドアテンション機構の実装

        相対位置エンコーディングとproximal biasをサポートしたアテンション計算を行います。
        Query, Key, Valueテンソルを複数のヘッドに分割し、並列にアテンションを計算します。
        相対位置エンコーディングを使用する場合、トークン間の相対的な距離情報を
        アテンションスコアに加算します。

        Args:
            query (torch.Tensor): クエリテンソル (B, C, T)
            key (torch.Tensor): キーテンソル (B, C, T)
            value (torch.Tensor): バリューテンソル (B, C, T)
            mask (Optional[torch.Tensor], optional): マスクテンソル

        Returns:
            tuple[torch.Tensor, torch.Tensor]: アテンションテンソルとアテンションマスク
        """
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = self._reshape_for_attention(query, b, t_t)
        key = self._reshape_for_attention(key, b, t_s)
        value = self._reshape_for_attention(value, b, t_s)

        # 基本的なアテンションスコアを計算 (scaled dot-product attention)
        scores = torch.matmul(
            query / math.sqrt(self.k_channels), key.transpose(-2, -1)
        )

        # 相対位置エンコーディングによるバイアスを追加
        # トークン間の相対的な距離情報をスコアに反映
        if self.window_size is not None:
            scores = self._add_relative_position_bias(scores, query, t_s, t_t)

        # Proximal biasを追加
        # 近接する位置への注意を促進するバイアス
        if self.proximal_bias:
            scores = self._add_proximal_bias(scores, t_s, t_t)

        # アテンションマスクを適用
        # パディング部分やブロック外の位置への注意を抑制
        if mask is not None:
            scores = self._apply_attention_mask(scores, mask, t_s, t_t)

        # ソフトマックスを適用して確率分布に変換
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)

        # アテンションを計算
        output = torch.matmul(p_attn, value)

        # 相対位置エンコーディングによるバイアスを追加
        if self.window_size is not None:
            output = self._add_relative_position_to_output(output, p_attn, t_s)

        # 出力を元の形状に戻す
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        相対位置重みとValue埋め込みの行列積を計算

        アテンション重みを相対位置表現に変換した後、
        Value用の相対位置埋め込みと掛け合わせて出力に加算します。

        Args:
            x (torch.Tensor): 相対位置アテンション重み [b, h, l, m]
                - b: バッチサイズ
                - h: ヘッド数
                - l: シーケンス長
                - m: 相対位置の範囲 (2*window_size+1)
            y (torch.Tensor): Value用相対位置埋め込み [h or 1, m, d]
                - h or 1: ヘッド数(共有時は1)
                - m: 相対位置の範囲
                - d: チャネル数

        Returns:
            torch.Tensor: 出力テンソル [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        クエリとKey用相対位置埋め込みの行列積を計算

        QueryとKey用の相対位置埋め込みを掛け合わせて、
        相対位置に基づくアテンションスコアを算出します。

        Args:
            x (torch.Tensor): クエリテンソル [b, h, l, d]
                - b: バッチサイズ
                - h: ヘッド数
                - l: シーケンス長
                - d: チャネル数
            y (torch.Tensor): Key用相対位置埋め込み [h or 1, m, d]
                - h or 1: ヘッド数(共有時は1)
                - m: 相対位置の範囲 (2*window_size+1)
                - d: チャネル数

        Returns:
            torch.Tensor: 相対位置アテンションスコア [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(
        self,
        relative_embeddings: torch.Tensor,
        length: int
    ) -> torch.Tensor:
        """
        シーケンス長に応じた相対位置埋め込みを取得

        固定サイズの相対位置埋め込みから、現在のシーケンス長に
        適したサブセットを抽出します。必要に応じてパディングを行います。

        Args:
            relative_embeddings (torch.Tensor):
                相対位置埋め込み [h or 1, 2*window_size+1, d]
            length (int): 現在のシーケンス長

        Returns:
            torch.Tensor: 抽出された相対位置埋め込み [h or 1, 2*length-1, d]
        """
        assert self.window_size is not None
        2 * self.window_size + 1  # type: ignore
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def s_relative_position_to_absolute_position(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        相対位置表現から絶対位置表現に変換

        相対位置で表現されたアテンションスコアを、
        通常の絶対位置表現に変換します。

        Args:
            x (torch.Tensor): 相対位置アテンションスコア [b, h, l, 2*l-1]
                - b: バッチサイズ
                - h: ヘッド数
                - l: シーケンス長
                - 2*l-1: 相対位置の範囲

        Returns:
            torch.Tensor: 絶対位置アテンションスコア [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1:
        ]
        return x_final

    def _absolute_position_to_relative_position(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        絶対位置表現から相対位置表現に変換

        通常の絶対位置表現のアテンション重みを、
        相対位置表現に変換します。

        Args:
            x (torch.Tensor): 絶対位置アテンション重み [b, h, l, l]
                - b: バッチサイズ
                - h: ヘッド数
                - l: シーケンス長

        Returns:
            torch.Tensor: 相対位置アテンション重み [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(
            x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]])
        )
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        """
        近接位置への注意を促すバイアスを生成

        位置間の距離に基づいて、近い位置ほど高いバイアス値を持つ
        行列を生成します。距離が離れるほど対数的に減衰します。

        Args:
            length (int): シーケンス長

        Returns:
            torch.Tensor: Proximal biasテンソル [1, 1, length, length]
                - 値は -log(1 + |i - j|) で計算される
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(
            torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0
        )


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ) -> None:
        """
        Feed-Forward Network (FFN)

        2層の畳み込み層とドロップアウトからなるフィードフォワードネットワーク。
        Transformerのアテンション層の後に適用され、非線形変換を行います。

        Args:
            in_channels (int): 入力チャネル数
            out_channels (int): 出力チャネル数
            filter_channels (int): 中間層のチャネル数
            kernel_size (int): 畳み込みカーネルサイズ
            p_dropout (float, optional): ドロップアウト率 (default: 0.0)
            activation (Optional[str], optional): 活性化関数
                - "gelu": GELU近似を使用
                - None: ReLUを使用 (default: None)
            causal (bool, optional): 因果的パディングを使用するか
                - True: 過去の情報のみを参照(自己回帰モデル用)
                - False: 双方向パディング (default: False)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        FFNの順伝播

        2層の畳み込みと非線形変換を適用します。

        Args:
            x (torch.Tensor): 入力テンソル (B, in_channels, T)
            x_mask (torch.Tensor): マスクテンソル (B, 1, T)

        Returns:
            torch.Tensor: 出力テンソル (B, out_channels, T)
        """
        # 第1層: in_channels -> filter_channels
        x = self.conv_1(self.padding(x * x_mask))
        # 活性化関数
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)  # GELU近似
        else:
            x = torch.relu(x)  # ReLU
        x = self.drop(x)
        # 第2層: filter_channels -> out_channels
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        因果的パディングを適用

        自己回帰モデル用に、過去の情報のみを参照するように
        左側のみにパディングを追加します。

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)

        Returns:
            torch.Tensor: パディング適用後のテンソル
        """
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1  # 左側のパディング
        pad_r = 0  # 右側はパディングなし
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        同じサイズを保つパディングを適用

        入力と出力の長さを同じに保つために、
        左右対称にパディングを追加します。

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)

        Returns:
            torch.Tensor: パディング適用後のテンソル
        """
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2  # 左側パディング
        pad_r = self.kernel_size // 2  # 右側パディング
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x
