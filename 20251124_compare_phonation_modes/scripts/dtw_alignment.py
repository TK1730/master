"""
DTW音声アライメントと差分可視化モジュール

このモジュールは2つの音声ファイルのメルスペクトログラムを計算し、
Dynamic Time Warping (DTW)を使用してアライメントを取り、
差分を計算して可視化する機能を提供します。
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


def load_audio(audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    音声ファイルを読み込む

    Args:
        audio_path: 音声ファイルのパス
        sr: サンプリングレート (デフォルト: 22050 Hz)

    Returns:
        音声データとサンプリングレートのタプル
    """
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr


def dynamic_range_compression(
    x: np.ndarray,
    clip_val: float = 1e-5
) -> np.ndarray:
    """
    対数スケールに変換（クリッピング付き）

    Args:
        x: 入力配列
        clip_val: クリッピングの最小値

    Returns:
        対数変換された配列
    """
    return np.log(np.clip(x, clip_val, None))


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    n_mels: int = 80,
    hop_length: int = 256
) -> np.ndarray:
    """
    メルスペクトログラムを計算する（analyze_features.pyの方法）

    Args:
        y: 音声データ
        sr: サンプリングレート
        n_fft: FFTウィンドウサイズ (デフォルト: 1024)
        n_mels: メル周波数ビンの数 (デフォルト: 80)
        hop_length: フレーム間のサンプル数 (デフォルト: 256)

    Returns:
        対数メルスペクトログラム (Time, n_mels) の形式で返す
    """
    # STFTで振幅スペクトルを計算
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    sp, _ = librosa.magphase(D)

    # メルフィルタバンクを作成
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=None
    )

    # メルスペクトルを計算
    mel_sp = np.dot(mel_basis, sp)

    # 対数変換
    log_mel = dynamic_range_compression(mel_sp)

    return log_mel.T  # (Time, n_mels) の形式で返す


def align_spectrograms_with_dtw(
    spec1: np.ndarray,
    spec2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DTWを使用して2つのスペクトログラムをアライメントする

    Args:
        spec1: 1つ目のスペクトログラム (Time, n_mels)
        spec2: 2つ目のスペクトログラム (Time, n_mels)

    Returns:
        DTWコストマトリックスとアライメントパスのタプル
    """
    # DTWを計算 (librosa.sequence.dtwは(Feature, Time)の形式を期待)
    D, wp = librosa.sequence.dtw(
        X=spec1.T,
        Y=spec2.T,
        metric='euclidean'
    )
    return D, wp


def compute_difference(
    spec1: np.ndarray,
    spec2: np.ndarray,
    alignment_path: np.ndarray
) -> np.ndarray:
    """
    アライメント後の2つのスペクトログラムの差分を計算する

    Args:
        spec1: 1つ目のスペクトログラム (Time, n_mels)
        spec2: 2つ目のスペクトログラム (Time, n_mels)
        alignment_path: DTWアライメントパス

    Returns:
        差分スペクトログラム (Time, n_mels)
    """
    # アライメントパスに基づいて差分を計算
    aligned_length = len(alignment_path)
    n_mels = spec1.shape[1]

    # アライメントされた差分を格納する配列
    diff_spec = np.zeros((aligned_length, n_mels))

    for i, (idx1, idx2) in enumerate(alignment_path):
        diff_spec[i, :] = spec1[idx1, :] - spec2[idx2, :]

    return diff_spec


def visualize_alignment_and_difference(
    spec1: np.ndarray,
    spec2: np.ndarray,
    diff_spec: np.ndarray,
    dtw_matrix: np.ndarray,
    alignment_path: np.ndarray,
    sr: int,
    hop_length: int,
    audio1_name: str,
    audio2_name: str,
    output_path: Optional[str] = None
) -> None:
    """
    スペクトログラム、DTWアライメント、差分を可視化する

    Args:
        spec1: 1つ目のスペクトログラム
        spec2: 2つ目のスペクトログラム
        diff_spec: 差分スペクトログラム
        dtw_matrix: DTWコストマトリックス
        alignment_path: DTWアライメントパス
        sr: サンプリングレート
        hop_length: ホップ長
        audio1_name: 1つ目の音声ファイル名
        audio2_name: 2つ目の音声ファイル名
        output_path: 保存先パス (Noneの場合は表示のみ)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1つ目のメルスペクトログラム
    img1 = librosa.display.specshow(
        spec1,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[0, 0],
        cmap='viridis'
    )
    axes[0, 0].set_title(f'Mel Spectrogram - {audio1_name}')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Mel Frequency')
    fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')

    # 2つ目のメルスペクトログラム
    img2 = librosa.display.specshow(
        spec2,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[0, 1],
        cmap='viridis'
    )
    axes[0, 1].set_title(f'Mel Spectrogram - {audio2_name}')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mel Frequency')
    fig.colorbar(img2, ax=axes[0, 1], format='%+2.0f dB')

    # DTWアライメントパス
    axes[1, 0].imshow(
        dtw_matrix.T,
        origin='lower',
        cmap='gray_r',
        aspect='auto'
    )
    axes[1, 0].plot(
        alignment_path[:, 0],
        alignment_path[:, 1],
        'r-',
        linewidth=2,
        label='Alignment Path'
    )
    axes[1, 0].set_title('DTW Cost Matrix and Alignment Path')
    axes[1, 0].set_xlabel(f'Frame Index - {audio1_name}')
    axes[1, 0].set_ylabel(f'Frame Index - {audio2_name}')
    axes[1, 0].legend()

    # 差分スペクトログラム
    img_diff = librosa.display.specshow(
        diff_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[1, 1],
        cmap='RdBu_r'
    )
    axes[1, 1].set_title('Difference Spectrogram (Aligned)')
    axes[1, 1].set_xlabel('Aligned Time')
    axes[1, 1].set_ylabel('Mel Frequency')
    fig.colorbar(img_diff, ax=axes[1, 1], format='%+2.0f dB')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {output_path}")
    else:
        plt.show()

    plt.close()


def compute_mse_difference(
    spec1: np.ndarray,
    spec2: np.ndarray,
    alignment_path: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    アライメント後の2つのスペクトログラムのMSE（平均二乗誤差）を計算する

    Args:
        spec1: 1つ目のスペクトログラム (Time, n_mels)
        spec2: 2つ目のスペクトログラム (Time, n_mels)
        alignment_path: DTWアライメントパス

    Returns:
        差分スペクトログラム（絶対値、Time, n_mels）とMSE値のタプル
    """
    # アライメントパスに基づいて差分を計算
    aligned_length = len(alignment_path)
    n_mels = spec1.shape[1]

    # アライメントされた差分を格納する配列
    diff_spec = np.zeros((aligned_length, n_mels))

    for i, (idx1, idx2) in enumerate(alignment_path):
        # 絶対値を取る
        diff_spec[i, :] = np.abs(spec1[idx1, :] - spec2[idx2, :])

    # MSEを計算
    mse = np.mean(diff_spec ** 2)

    return diff_spec, mse


def visualize_difference_only(
    diff_spec: np.ndarray,
    sr: int,
    hop_length: int,
    audio1_name: str,
    audio2_name: str,
    mse: float,
    output_path: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    差分スペクトログラムのみを可視化する

    Args:
        diff_spec: 差分スペクトログラム（絶対値、Time, n_mels）
        sr: サンプリングレート
        hop_length: ホップ長
        audio1_name: 1つ目の音声ファイル名
        audio2_name: 2つ目の音声ファイル名
        mse: 平均二乗誤差
        output_path: 保存先パス (Noneの場合は表示のみ)
        cmap: カラーマップ (デフォルト: 'viridis')
        figsize: 図のサイズ (デフォルト: (12, 6))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 差分の最大値を取得（絶対値なのでvmin=0）
    vmax = diff_spec.max()
    vmin = 0

    # 差分スペクトログラムを表示 (librosa.display.specshowは(n_mels, Time)を期待)
    img_diff = librosa.display.specshow(
        diff_spec.T,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )

    # タイトルと軸ラベル
    ax.set_title(
        f'Absolute Difference Spectrogram: '
        f'|{audio1_name} - {audio2_name}|\n'
        f'MSE: {mse:.4f}',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mel Frequency (Hz)', fontsize=12)

    # カラーバーを追加
    cbar = fig.colorbar(img_diff, ax=ax, format='%+2.3f')
    cbar.set_label('Absolute Difference (log scale)', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"差分スペクトログラムを保存しました: {output_path}")
    else:
        plt.show()

    plt.close()


def compare_audio_with_dtw(
    audio_path1: str,
    audio_path2: str,
    output_path: Optional[str] = None,
    sr: int = 22050,
    n_mels: int = 128,
    hop_length: int = 512
) -> None:
    """
    2つの音声ファイルをDTWでアライメントし、差分を可視化する

    Args:
        audio_path1: 1つ目の音声ファイルのパス
        audio_path2: 2つ目の音声ファイルのパス
        output_path: 保存先パス (Noneの場合は表示のみ)
        sr: サンプリングレート (デフォルト: 22050 Hz)
        n_mels: メル周波数ビンの数 (デフォルト: 128)
        hop_length: フレーム間のサンプル数 (デフォルト: 512)
    """
    print(f"音声ファイル1を読み込み中: {audio_path1}")
    y1, sr1 = load_audio(audio_path1, sr=sr)

    print(f"音声ファイル2を読み込み中: {audio_path2}")
    y2, sr2 = load_audio(audio_path2, sr=sr)

    print("メルスペクトログラムを計算中...")
    spec1 = compute_mel_spectrogram(
        y1, sr1, n_mels=n_mels, hop_length=hop_length
    )
    spec2 = compute_mel_spectrogram(
        y2, sr2, n_mels=n_mels, hop_length=hop_length
    )

    print("DTWアライメントを実行中...")
    dtw_matrix, alignment_path = align_spectrograms_with_dtw(spec1, spec2)

    print("差分を計算中...")
    diff_spec = compute_difference(spec1, spec2, alignment_path)

    print("可視化中...")
    audio1_name = Path(audio_path1).stem
    audio2_name = Path(audio_path2).stem

    visualize_alignment_and_difference(
        spec1,
        spec2,
        diff_spec,
        dtw_matrix,
        alignment_path,
        sr,
        hop_length,
        audio1_name,
        audio2_name,
        output_path
    )

    print("処理完了!")
