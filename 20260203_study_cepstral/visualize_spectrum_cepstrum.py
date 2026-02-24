"""
パワースペクトルと実ケプストラムの可視化機能

音声信号からパワースペクトルと実ケプストラムを計算し、
グラフとして可視化する機能を提供します。
"""

from pathlib import Path
from cepstrum_analysis import (
    calculate_real_cepstrum,
    calculate_spectral_envelope,
    plot_power_spectrum,
    plot_real_cepstrum
)
import numpy as np


def visualize_power_spectrum_and_cepstrum(
        audio_path,
        frame_length=2048,
        hop_length=512,
        save_dir=None):
    """
    パワースペクトルと実ケプストラムを可視化する

    Parameters
    ----------
    audio_path : str or Path
        音声ファイルのパス
    frame_length : int, optional
        フレーム長 (default: 2048)
    hop_length : int, optional
        ホップサイズ (default: 512)
    save_dir : str or Path, optional
        保存先ディレクトリ (default: None)

    Returns
    -------
    power_spectrum : np.ndarray
        パワースペクトル配列
    real_cepstrum : np.ndarray
        実ケプストラム配列
    sr : int
        サンプリングレート
    """
    print(f"\n音声ファイルを処理中: {audio_path}")

    # 実ケプストラムを計算（パワースペクトルも内部で計算される）
    real_cepstrum, quefrency, sr = calculate_real_cepstrum(
        audio_path,
        frame_length=frame_length,
        hop_length=hop_length
    )

    print(f"サンプリングレート: {sr} Hz")
    print(f"実ケプストラム形状: {real_cepstrum.shape}")

    # パワースペクトルを再計算（可視化用）
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    frames = librosa.util.frame(
        y,
        frame_length=frame_length,
        hop_length=hop_length
    )
    spectrum = np.fft.rfft(frames, axis=0)
    power_spectrum = np.abs(spectrum) ** 2

    print(f"パワースペクトル形状: {power_spectrum.shape}")

    # スペクトル包絡を計算
    print("スペクトル包絡を計算中...")
    spectral_envelope = calculate_spectral_envelope(
        power_spectrum,
        sr,
        lifter_order=30
    )
    print(f"スペクトル包絡形状: {spectral_envelope.shape}")

    # 保存先を設定
    power_spectrum_save_path = None
    cepstrum_save_path = None

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        audio_name = Path(audio_path).stem

        power_spectrum_save_path = (
            save_path / f"{audio_name}_power_spectrum.png"
        )
        cepstrum_save_path = save_path / f"{audio_name}_real_cepstrum.png"

    # パワースペクトルを可視化（スペクトル包絡を重ねて描画）
    print("\nパワースペクトルを可視化中...")
    plot_power_spectrum(
        power_spectrum,
        sr,
        frame_length=frame_length,
        save_path=power_spectrum_save_path,
        spectral_envelope=spectral_envelope
    )

    # 実ケプストラムを可視化
    print("実ケプストラムを可視化中...")
    plot_real_cepstrum(
        real_cepstrum,
        sr,
        save_path=cepstrum_save_path
    )

    return power_spectrum, real_cepstrum, sr
