"""
ケプストラム分析モジュール

音声信号からケプストラム（cepstrum）を計算する機能を提供します。
ケプストラムは音声のスペクトル包絡とピッチ情報を分離するために使用されます。
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_cepstrum(audio_path, n_fft=2048, hop_length=512):
    """
    音声ファイルからケプストラムを計算する

    Parameters
    ----------
    audio_path : str or Path
        音声ファイルのパス
    n_fft : int, optional
        FFTのウィンドウサイズ (default: 2048)
    hop_length : int, optional
        フレーム間のホップサイズ (default: 512)

    Returns
    -------
    cepstrum : np.ndarray
        ケプストラム配列 (frames x quefrency_bins)
    quefrency : np.ndarray
        ケフレンシー軸（時間軸）
    sr : int
        サンプリングレート
    """
    # 音声ファイルを読み込む
    y, sr = librosa.load(audio_path, sr=None)

    # 短時間フーリエ変換（STFT）を計算
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # パワースペクトルを計算
    power_spectrum = np.abs(stft) ** 2

    # 対数パワースペクトルを計算（小さい値でクリッピング）
    log_power_spectrum = np.log(power_spectrum + 1e-10)

    # 逆フーリエ変換でケプストラムを計算
    cepstrum = np.fft.ifft(log_power_spectrum, axis=0).real

    # ケフレンシー軸を作成（サンプル単位）
    quefrency = np.arange(cepstrum.shape[0]) / sr

    return cepstrum, quefrency, sr


def calculate_real_cepstrum(audio_path, frame_length=1024, hop_length=256):
    """
    実ケプストラム（Real Cepstrum）を計算する

    Parameters
    ----------
    audio_path : str or Path
        音声ファイルのパス
    frame_length : int, optional
        フレーム長 (default: 2048)
    hop_length : int, optional
        ホップサイズ (default: 512)

    Returns
    -------
    real_cepstrum : np.ndarray
        実ケプストラム配列
    quefrency : np.ndarray
        ケフレンシー軸
    sr : int
        サンプリングレート
    """
    # 音声ファイルを読み込む
    y, sr = librosa.load(audio_path, sr=None)

    # フレーム分割
    frames = librosa.util.frame(
        y,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # 各フレームでFFTを計算
    spectrum = np.fft.rfft(frames, axis=0)

    # パワースペクトル
    power_spectrum = np.abs(spectrum) ** 2
    
    # 対数を取る
    log_spectrum = np.log(power_spectrum + 1e-10)

    # 逆FFTでケプストラムを計算
    real_cepstrum = np.fft.irfft(log_spectrum, axis=0)

    # ケフレンシー軸
    quefrency = np.arange(real_cepstrum.shape[0]) / sr

    return real_cepstrum, quefrency, sr


def calculate_spectral_envelope(power_spectrum, sr, lifter_order=30):
    """
    パワースペクトルからスペクトル包絡を計算する

    ケプストラムを使用してスペクトル包絡を抽出します。
    低次のケプストラム係数のみを保持することで、
    細かいハーモニック構造を除去し、滑らかな包絡線を得ます。

    Parameters
    ----------
    power_spectrum : np.ndarray
        パワースペクトル配列 (frequency_bins x frames)
    sr : int
        サンプリングレート
    lifter_order : int, optional
        リフター次数（この次数以下のケプストラム係数を保持） (default: 30)

    Returns
    -------
    spectral_envelope : np.ndarray
        スペクトル包絡 (frequency_bins x frames)
    """
    # 対数パワースペクトルを計算
    log_power_spectrum = np.log(power_spectrum + 1e-10)

    # 逆FFTでケプストラムを計算
    cepstrum = np.fft.irfft(log_power_spectrum, axis=0)

    # リフタリング: 低次のケプストラム係数のみを保持
    liftered_cepstrum = np.zeros_like(cepstrum)
    liftered_cepstrum[:lifter_order, :] = cepstrum[:lifter_order, :]

    # FFTでスペクトル包絡に変換
    log_envelope = np.fft.rfft(liftered_cepstrum, axis=0)

    # 指数を取ってパワー領域に戻す
    spectral_envelope = np.exp(log_envelope.real)

    return spectral_envelope


def plot_power_spectrum(power_spectrum, sr, frame_length=2048,
                        save_path=None, spectral_envelope=None):
    """
    パワースペクトルを可視化する

    Parameters
    ----------
    power_spectrum : np.ndarray
        パワースペクトル配列 (frequency_bins x frames)
    sr : int
        サンプリングレート
    frame_length : int, optional
        フレーム長 (default: 2048)
    save_path : str or Path, optional
        保存先パス (default: None)
    spectral_envelope : np.ndarray, optional
        スペクトル包絡（重ねて描画する場合） (default: None)
    """
    # 周波数軸を作成
    freqs = np.fft.rfftfreq(frame_length, 1/sr)

    # 最初のフレームのパワースペクトルを取得
    first_frame_spectrum = power_spectrum[:, 0]
    
    # プロット作成
    plt.figure(figsize=(12, 8))
    
    # 1つ目のサブプロット: 最初のフレームのパワースペクトル
    plt.subplot(2, 1, 1)
    plt.plot(freqs, 10 * np.log10(first_frame_spectrum + 1e-10),
             label='Power Spectrum', alpha=0.7)

    # スペクトル包絡を重ねて描画
    if spectral_envelope is not None:
        first_frame_envelope = spectral_envelope[:, 0]
        plt.plot(freqs, 10 * np.log10(first_frame_envelope + 1e-10),
                 'r-', linewidth=2, label='Spectral Envelope')
        plt.legend()

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Power Spectrum with Spectral Envelope (First Frame)')
    plt.grid(True)
    plt.xlim([0, sr/2])
    
    # 2つ目のサブプロット: 全フレームのパワースペクトル(スペクトログラム)
    plt.subplot(2, 1, 2)
    plt.imshow(
        10 * np.log10(power_spectrum + 1e-10),
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, power_spectrum.shape[1], 0, sr/2]
    )
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.title('Power Spectrogram')
    plt.colorbar(label='Power (dB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"パワースペクトル画像を保存しました: {save_path}")
    
    plt.show()


def plot_real_cepstrum(real_cepstrum, sr, max_quefrency=0.02,
                       save_path=None):
    """
    実ケプストラムを可視化する

    Parameters
    ----------
    real_cepstrum : np.ndarray
        実ケプストラム配列 (quefrency_bins x frames)
    sr : int
        サンプリングレート
    max_quefrency : float, optional
        表示する最大ケフレンシー（秒） (default: 0.02)
    save_path : str or Path, optional
        保存先パス (default: None)
    """
    # ケフレンシー軸を作成
    quefrency = np.arange(real_cepstrum.shape[0]) / sr
    
    # 最初のフレームのケプストラムを取得
    first_frame_cepstrum = real_cepstrum[:, 0]
    
    # 表示範囲を制限
    max_idx = int(max_quefrency * sr)
    quefrency_plot = quefrency[:max_idx]
    cepstrum_plot = first_frame_cepstrum[:max_idx]
    
    # プロット作成
    plt.figure(figsize=(12, 8))
    
    # 1つ目のサブプロット: 最初のフレームの実ケプストラム
    plt.subplot(2, 1, 1)
    plt.plot(quefrency_plot * 1000, cepstrum_plot)
    plt.xlabel('Quefrency (ms)')
    plt.ylabel('Amplitude')
    plt.title('Real Cepstrum (First Frame)')
    plt.grid(True)
    
    # 2つ目のサブプロット: 全フレームの実ケプストラム(ケプストログラム)
    plt.subplot(2, 1, 2)
    plt.imshow(
        real_cepstrum[:max_idx, :],
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, real_cepstrum.shape[1], 0, max_quefrency * 1000]
    )
    plt.xlabel('Frame')
    plt.ylabel('Quefrency (ms)')
    plt.title('Real Cepstrogram')
    plt.colorbar(label='Amplitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"実ケプストラム画像を保存しました: {save_path}")
    
    plt.show()


def plot_cepstrum(cepstrum, quefrency, sr, max_quefrency=0.02,
                  save_path=None):
    """
    ケプストラムを可視化する

    Parameters
    ----------
    cepstrum : np.ndarray
        ケプストラム配列
    quefrency : np.ndarray
        ケフレンシー軸
    sr : int
        サンプリングレート
    max_quefrency : float, optional
        表示する最大ケフレンシー（秒） (default: 0.02)
    save_path : str or Path, optional
        保存先パス (default: None)
    """
    # 最初のフレームのケプストラムを取得
    first_frame_cepstrum = cepstrum[1:, 0]

    # 表示範囲を制限
    max_idx = int(max_quefrency * sr)
    quefrency_plot = quefrency[:max_idx]
    cepstrum_plot = first_frame_cepstrum[:max_idx]

    # プロット作成
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(quefrency_plot * 1000, cepstrum_plot)
    plt.xlabel('Quefrency (ms)')
    plt.ylabel('Amplitude')
    plt.title('Cepstrum (First Frame)')
    plt.grid(True)

    # 全フレームのケプストラムを2D表示
    plt.subplot(2, 1, 2)
    plt.imshow(
        cepstrum[:max_idx, :],
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, cepstrum.shape[1], 0, max_quefrency * 1000]
    )
    plt.xlabel('Frame')
    plt.ylabel('Quefrency (ms)')
    plt.title('Cepstrogram')
    plt.colorbar(label='Amplitude')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ケプストラム画像を保存しました: {save_path}")

    plt.show()


def extract_pitch_from_cepstrum(cepstrum, sr, min_f0=80, max_f0=400):
    """
    ケプストラムからピッチ（基本周波数）を推定する

    Parameters
    ----------
    cepstrum : np.ndarray
        ケプストラム配列
    sr : int
        サンプリングレート
    min_f0 : float, optional
        最小基本周波数 (Hz) (default: 80)
    max_f0 : float, optional
        最大基本周波数 (Hz) (default: 400)

    Returns
    -------
    f0_estimates : np.ndarray
        各フレームの推定基本周波数 (Hz)
    """
    # ケフレンシー範囲を周波数範囲に変換
    min_quefrency = 1.0 / max_f0
    max_quefrency = 1.0 / min_f0

    # サンプル数に変換
    min_idx = int(min_quefrency * sr)
    max_idx = int(max_quefrency * sr)

    # 各フレームでピークを探索
    f0_estimates = []
    for frame_idx in range(cepstrum.shape[1]):
        frame_cepstrum = cepstrum[min_idx:max_idx, frame_idx]

        # ピークのインデックスを検出
        peak_idx = np.argmax(frame_cepstrum)
        actual_idx = peak_idx + min_idx

        # ケフレンシーから周波数に変換
        quefrency = actual_idx / sr
        if quefrency > 0:
            f0 = 1.0 / quefrency
        else:
            f0 = 0

        f0_estimates.append(f0)

    return np.array(f0_estimates)


def process_audio_dataset(dataset_dir, output_dir, pattern="**/*.wav"):
    """
    データセット内の音声ファイルを一括処理してケプストラムを計算

    Parameters
    ----------
    dataset_dir : str or Path
        データセットディレクトリのパス
    output_dir : str or Path
        出力ディレクトリのパス
    pattern : str, optional
        検索パターン (default: "**/*.wav")
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 音声ファイルを検索
    audio_files = list(dataset_path.glob(pattern))

    print(f"見つかった音声ファイル数: {len(audio_files)}")

    # 各ファイルを処理
    for idx, audio_file in enumerate(audio_files):
        print(f"\n処理中 ({idx+1}/{len(audio_files)}): {audio_file.name}")

        try:
            # ケプストラムを計算
            cepstrum, quefrency, sr = calculate_cepstrum(audio_file)

            # 相対パスを取得
            rel_path = audio_file.relative_to(dataset_path)
            save_dir = output_path / rel_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # 画像として保存
            save_path = save_dir / f"{audio_file.stem}_cepstrum.png"
            plot_cepstrum(
                cepstrum,
                quefrency,
                sr,
                save_path=save_path
            )

            # NumPy配列として保存
            npy_path = save_dir / f"{audio_file.stem}_cepstrum.npy"
            np.save(npy_path, cepstrum)

            # ピッチ推定
            f0_estimates = extract_pitch_from_cepstrum(cepstrum, sr)
            f0_path = save_dir / f"{audio_file.stem}_f0.npy"
            np.save(f0_path, f0_estimates)

            print(f"  - 画像保存: {save_path}")
            print(f"  - データ保存: {npy_path}")
            print(f"  - F0保存: {f0_path}")

        except Exception as e:
            print(f"  エラー: {e}")
            continue

    print(f"\n処理完了! 合計 {len(audio_files)} ファイル")
