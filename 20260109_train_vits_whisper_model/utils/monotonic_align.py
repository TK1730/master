import math
from typing import Any
import torch
import torch.nn.functional as F
import numba
from numba import cuda
from numpy import float32, int32, zeros
import numpy as np

from utils.model import sequence_mask, convert_pad_shape


"""
Monotonic Alignment Search (MAS) for VITS2

このモジュールはVITS2モデルで使用されるMonotonic Alignment Search (MAS)を
実装しています。MASはテキストと音声の最適なアライメントを見つけるための
アルゴリズムです。

実装方式:
    - CPU版: Numba JITコンパイラを使用した高速化
        - 利点: 追加の依存関係が不要
        - 欠点: バッチサイズが大きいと遅い
        - 適している状況: 小規模バッチ、CPU環境、推論時

    - GPU版: CUDAカーネルを使用
        - 利点: 大規模バッチで高速
        - 欠点: CUDA環境が必要、追加の設定が必要
        - 適している状況: 大規模バッチ、GPU学習時

現在の実装: CPU版(Numba JIT)を使用
注意: GPU版を使用する場合は、`maximum_path`を`maximum_path_cuda`に
     置き換えてください。
"""


# * Ready and Tested
def search_path(
    z_p: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    x_mask: torch.Tensor,
    y_mask: torch.Tensor,
    mas_noise_scale: float = 0.01,
    use_mas_noise_scale: bool = True,
):
    """
    Monotonic Alignment Search (MAS)

    テキストと音声の間の最適なモノトニックアライメントを探索します。
    ガウス分布の対数尤度を計算し、動的計画法で最適パスを見つけます。

    Args:
        z_p (torch.Tensor): Flowで変換後の潜在変数
            Shape: [batch, latent_dim, spec_len]
        m_p (torch.Tensor): テキストからの平均
            Shape: [batch, latent_dim, text_len]
        logs_p (torch.Tensor): テキストからの対数分散
            Shape: [batch, latent_dim, text_len]
        x_mask (torch.Tensor): テキストのマスク
            Shape: [batch, 1, text_len]
        y_mask (torch.Tensor): スペクトログラムのマスク
            Shape: [batch, 1, spec_len]
        mas_noise_scale (float): noise scale MAS. Defaults to 0.01.
        use_mas_noise_scale (bool): noise scale MASを使用するか.
            Defaults to True.

    Returns:
        torch.Tensor: アライメント行列
            Shape: [batch, 1, spec_len, text_len]
    """
    with torch.no_grad():
        # ガウス分布の対数尤度を計算
        # log p(z_p | m_p, logs_p) = -1/2 * log(2π) - logσ - 1/2 * ((z-μ)/σ)^2

        o_scale = torch.exp(-2 * logs_p)
        # -> 1/σ^2: [batch, latent_dim, text_len]

        # 各項を分けて計算
        logp1 = torch.sum(
            -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
        )  # [batch, 1, text_len]

        logp2 = torch.matmul(
            -0.5 * (z_p**2).mT, o_scale
        )  # [batch, spec_len, latent_dim] @ [batch, latent_dim, text_len]
        #    = [batch, spec_len, text_len]

        logp3 = torch.matmul(
            z_p.mT, (m_p * o_scale)
        )  # [batch, spec_len, latent_dim] @ [batch, latent_dim, text_len]
        #    = [batch, spec_len, text_len]

        logp4 = torch.sum(
            -0.5 * (m_p**2) * o_scale, [1], keepdim=True
        )  # [batch, 1, text_len]

        # 合計: 対数尤度
        logp = logp1 + logp2 + logp3 + logp4  # [batch, spec_len, text_len]

        # ノイズスケールMAS: 学習を安定化するためのノイズ
        if use_mas_noise_scale:
            epsilon = (
                torch.std(logp)
                * torch.randn_like(logp)
                * mas_noise_scale
            )
            logp = logp + epsilon

        # アテンションマスクを作成
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(
            y_mask, -1
        )  # [batch, 1, text_len] * [batch, spec_len, 1]
        #    = [batch, spec_len, text_len]

        # 動的計画法で最適パスを探索
        # 注意: 現在はCPU版(Numba JIT)を使用
        # GPU版を使用する場合は`maximum_path_cuda`に置き換える
        attn = (
            maximum_path(logp, attn_mask.squeeze(1))
            .unsqueeze(1)
            .detach()  # [batch, 1, spec_len, text_len]
        )
    return attn


def generate_path(
    duration: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    持続時間からアライメントパスを生成します。

    推論時に使用され、予測された持続時間からハードアテンションパスを
    生成します。

    Args:
        duration (torch.Tensor): 各音素の持続時間
            Shape: [batch, 1, text_len]
        mask (torch.Tensor): マスク
            Shape: [batch, 1, spec_len, text_len]

    Returns:
        torch.Tensor: アライメントパス
            Shape: [batch, 1, spec_len, text_len]
    """
    b, _, t_y, t_x = mask.shape

    # 累積持続時間を計算
    cum_duration = torch.cumsum(duration, -1)

    # 各音素の終了位置を取得
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    # 前の音素の累積を引いて、各音素のセグメントを取得
    path = path - F.pad(path, convert_pad_shape(
        [[0, 0], [1, 0], [0, 0]]
    ))[:, :-1]

    # 正しい形状に変換してマスクを適用
    path = path.unsqueeze(1).mT * mask
    return path


def maximum_path(
    neg_cent: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    動的計画法で最大パスを計算します (CPU版: Numba JIT)

    モノトニック制約の下で、負のコストを最大化するパスを見つけます。

    実装方式:
        - CPU上でNumba JITコンパイラを使用して高速化
        - データを一度NumPyに変換してCPUで処理
        - 結果をTorchテンソルに戻す

    性能特性:
        - 小規模バッチ (batch_size < 8): 高速
        - 中規模バッチ (8 <= batch_size <= 32): 適度
        - 大規模バッチ (batch_size > 32): GPU版の使用を推奨

    GPU版への切り替え:
        大規模バッチで高速化が必要な場合は、CUDAカーネル版を
        使用してください。該当コードはファイル内にコメントアウトされて
        います。

    Args:
        neg_cent (torch.Tensor): 負のコスト行列 (対数尤度)
            Shape: [batch, spec_len, text_len]
        mask (torch.Tensor): マスク
            Shape: [batch, spec_len, text_len]

    Returns:
        torch.Tensor: アライメントパス
            Shape: [batch, spec_len, text_len]
    """
    # 元のデバイスとdtypeを保存
    device = neg_cent.device
    dtype = neg_cent.dtype

    # NumPy配列に変換 (CPU上で処理)
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent.shape, dtype=int32)

    # 各系列の有効長を取得
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)

    # Numba JITでコンパイルされた関数で最大パスを計算
    __maximum_path_jit(path, neg_cent, t_t_max, t_s_max)

    # Torchテンソルに戻して元のデバイスに配置
    return torch.from_numpy(path).to(device=device, dtype=dtype)


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore
def __maximum_path_jit(paths: Any, values: Any, t_ys: Any, t_xs: Any) -> None:
    """
    Numba JITで最大パスを計算します。

    動的計画法を使用して、モノトニック制約の下で最適パスを見つけます。

    アルゴリズム:
        1. 前向きパス: 動的計画法で各位置の最大値を計算
        2. 後ろ向きトレース: 最適パスを逆向きにたどる

    制約:
        - モノトニック: xとyの両方が単調増加
        - x == yの位置は使用不可 (同じフレームに複数の音素を割り当てない)

    Args:
        paths (ndarray): 出力パスを格納する配列
            Shape: [batch, spec_len, text_len]
        values (ndarray): コスト値 (対数尤度)
            Shape: [batch, spec_len, text_len]
        t_ys (ndarray): 各バッチの有効なスペクトログラム長
            Shape: [batch]
        t_xs (ndarray): 各バッチの有効なテキスト長
            Shape: [batch]
    """
    b = paths.shape[0]
    max_neg_val = -1e9  # 無効なパスに割り当てる値

    # 各バッチを順に処理
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]  # スペクトログラムの有効長
        t_x = t_xs[i]  # テキストの有効長

        v_prev = v_cur = 0.0
        index = t_x - 1  # 最後のテキスト位置から開始

        # 前向きパス: 動的計画法で最大値を計算
        for y in range(t_y):
            # モノトニック制約: xは[max(0, t_x+y-t_y), min(t_x, y+1))の範囲
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                # 前の位置からの遷移コストを計算
                if x == y:
                    v_cur = max_neg_val  # 同じ位置は無効
                else:
                    v_cur = value[y - 1, x]  # 上からの遷移

                if x == 0:
                    if y == 0:
                        v_prev = 0.0  # 開始位置
                    else:
                        v_prev = max_neg_val  # 無効
                else:
                    v_prev = value[y - 1, x - 1]  # 左上からの遷移

                # 現在の位置の最大値を更新
                value[y, x] += max(v_prev, v_cur)

        # 後ろ向きトレース: 最適パスを構築
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1  # この位置をパスに含める
            # 次の位置を決定
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1


# ==================== GPU版 (CUDA Implementation) ====================


def maximum_path_cuda(
    neg_cent: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    動的計画法で最大パスを計算します (GPU版: CUDA)

    GPU上でCUDAカーネルを使用して並列処理を行います。
    大規模バッチで高速な処理が可能です。

    実装方式:
        - CUDA環境が必要
        - 各バッチを並列に処理
        - スレッド数はシーケンス長に応じて自動調整

    性能特性:
        - 小規模バッチ (<8): CPU版の方が速い可能性あり
        - 中規模バッチ (8-32): GPU版が高速化開始
        - 大規模バッチ (>32): GPU版が大幅に高速

    使用方法:
        search_path関数内の`maximum_path`を`maximum_path_cuda`に
        置き換えてください。

    注意事項:
        - CUDA対応GPUが必要
        - numba.cudaが正しく設定されている必要がある
        - threadsperblockとblockspergridの値は環境に応じて調整可能

    Args:
        neg_cent (torch.Tensor): 負のコスト行列 (対数尤度)
            Shape: [batch, spec_len, text_len]
        mask (torch.Tensor): マスク
            Shape: [batch, spec_len, text_len]

    Returns:
        torch.Tensor: アライメントパス
            Shape: [batch, spec_len, text_len]
    """
    # 元のデバイスとdtypeを保存
    device = neg_cent.device
    dtype = neg_cent.dtype

    # CUDA配列に変換
    neg_cent_device = cuda.as_cuda_array(neg_cent)
    path_device = cuda.device_array(neg_cent.shape, dtype=np.int32)
    t_t_max_device = cuda.as_cuda_array(mask.sum(1, dtype=torch.int32)[:, 0])
    t_s_max_device = cuda.as_cuda_array(mask.sum(2, dtype=torch.int32)[:, 0])

    # スレッド構成を設定
    # 各バッチを1つの並列ユニット(i = cuda.grid(1))として処理
    # 注意: 現状のカーネル実装は内部でループを回しているため、スレッド並列は不要
    threadsperblock = 1
    blockspergrid = neg_cent.shape[0]

    # CUDAカーネルを実行
    maximum_path_cuda_jit[blockspergrid, threadsperblock](
        path_device, neg_cent_device, t_t_max_device, t_s_max_device
    )

    # デバイス配列をホストにコピーしてTensorに変換
    path = torch.as_tensor(
        path_device.copy_to_host(), device=device, dtype=dtype
    )
    return path


# CUDA 関数は CUDA が利用可能な場合のみ定義
try:
    @cuda.jit("void(int32[:,:,:], float32[:,:,:], int32[:], int32[:])")
    def maximum_path_cuda_jit(
        paths: Any,
        values: Any,
        t_ys: Any,
        t_xs: Any,
    ) -> None:
        """
        CUDAカーネルで最大パスを計算します。

        GPU上で並列に動的計画法を実行します。各バッチが1つのブロックで
        処理され、ブロック内のスレッドが協調して計算を行います。

        CUDA実装の特徴:
            - 各バッチを独立に並列処理
            - ブロック内同期でスレッド間の協調を保証
            - 共有メモリは使用せず、グローバルメモリを直接操作

        スレッド構成:
            - blockIdx.x: バッチインデックス (0 ~ batch_size-1)
            - threadIdx.x: スレッドインデックス (0 ~ max(spec_len, text_len)-1)

        制約:
            - モノトニック: xとyの両方が単調増加
            - x == yの位置は使用不可

        最適化の余地:
            - 共有メモリを使用してグローバルメモリアクセスを削減
            - スレッドブロックサイズの動的調整
            - ウォープレベルの最適化

        Args:
            paths (cuda.devicearray): 出力パスを格納するデバイス配列
                Shape: [batch, spec_len, text_len]
            values (cuda.devicearray): コスト値 (対数尤度)
                Shape: [batch, spec_len, text_len]
            t_ys (cuda.devicearray): 各バッチの有効なスペクトログラム長
                Shape: [batch]
            t_xs (cuda.devicearray): 各バッチの有効なテキスト長
                Shape: [batch]
        """
        max_neg_val = -1e9  # 無効なパスに割り当てる値

        # 現在のスレッドが処理するバッチインデックス
        i = cuda.grid(1)

        # インデックスが範囲外の場合は終了
        if i >= paths.shape[0]:
            return

        # 現在のバッチのデータを取得
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        # 前向きパス: 動的計画法で最大値を計算
        for y in range(t_y):
            # モノトニック制約の範囲を計算
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                # 前の位置からの遷移コストを計算
                if x == y:
                    v_cur = max_neg_val  # 同じ位置は無効
                else:
                    v_cur = value[y - 1, x]  # 上からの遷移

                if x == 0:
                    if y == 0:
                        v_prev = 0.0  # 開始位置
                    else:
                        v_prev = max_neg_val  # 無効
                else:
                    v_prev = value[y - 1, x - 1]  # 左上からの遷移

                # 現在の位置の最大値を更新
                value[y, x] += max(v_prev, v_cur)

        # 後ろ向きトレース: 最適パスを構築
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1  # この位置をパスに含める
            # 次の位置を決定
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1

        # スレッド同期: 全てのスレッドの処理完了を待つ
        cuda.syncthreads()

    # CUDA関数が正常に定義されたことを示すフラグ
    _CUDA_AVAILABLE = True

except Exception as e:
    # CUDAが利用できない場合は警告を出してスキップ
    import warnings
    warnings.warn(
        f"CUDA is not available for monotonic_align. "
        f"Using CPU version (Numba JIT) only. Error: {e}",
        RuntimeWarning
    )
    _CUDA_AVAILABLE = False

    # ダミー関数を定義してエラーを防ぐ
    def maximum_path_cuda_jit(*args, **kwargs):
        raise RuntimeError(
            "CUDA version of maximum_path is not available. "
            "Please use the CPU version (maximum_path) instead."
        )
