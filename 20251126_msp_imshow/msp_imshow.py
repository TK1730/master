from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from utils import config, functions

voice_path = Path(
    './dataset/preprocessed/jvs_ver1/nonpara30w_ver2/'
    'jvs001/wav/BASIC5000_0235.wav'
    )
pseudo_path = Path(
    './dataset/whisper_using_vits/'
    'jvs001/wav/BASIC5000_0235.wav'
    )
whisper_path = Path(
    './dataset/preprocessed/'
    'jvs_ver1/whisper10/jvs001/wav/BASIC5000_0235.wav'
    )


def load_and_extract_melspec(audio_path, sr=22050, n_mels=80):
    """音声ファイルを読み込んでメルスペクトログラムを抽出"""
    y, _ = librosa.load(audio_path, sr=sr)
    D = librosa.stft(
        y,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        pad_mode='reflect'
    ).T
    sp, phase = librosa.magphase(D)
    msp = np.matmul(sp, config.mel_filter.T)
    lmsp = functions.dynamic_range_compression(msp)
    return lmsp


def plot_mel_spectrograms(voice_path, pseudo_path, whisper_path):
    """3つのメルスペクトログラムを個別に描画"""
    # メルスペクトログラムを抽出
    voice_mel = load_and_extract_melspec(voice_path).T
    pseudo_mel = load_and_extract_melspec(pseudo_path).T
    whisper_mel = load_and_extract_melspec(whisper_path).T
    print(voice_mel.shape, pseudo_mel.shape, whisper_mel.shape)
    # Voice (nonpara30)を描画
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(
        voice_mel[:, 30:210], x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length)
    plt.ylabel('Mel Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('pseudo (nonpara30w_ver2)', fontsize=14, fontweight='bold')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    # Pseudo (nonpara30w)を描画
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(
        pseudo_mel[:, 30:210], x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length)
    plt.ylabel('Mel Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Pseudo (using vits)', fontsize=14, fontweight='bold')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    # Whisper (whisper10)を描画
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(
        whisper_mel[:, 30:210], x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length)
    plt.ylabel('Mel Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Whisper (whisper10)', fontsize=14, fontweight='bold')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    return voice_mel, pseudo_mel, whisper_mel


# メルスペクトログラムを描画
if __name__ == "__main__":
    voice_mel, pseudo_mel, whisper_mel = plot_mel_spectrograms(
        voice_path, pseudo_path, whisper_path
    )
