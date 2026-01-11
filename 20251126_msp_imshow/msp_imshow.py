from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

sys.path.append('..')


from utils import config, functions

# Base Directories
VOICE_BASE = Path('../dataset/preprocessed/jvs_ver1/nonpara30w_ver2/')
PSEUDO_BASE = Path('../dataset/pseudo_whisper_vits/whisper_converted_v2')
WHISPER_BASE = Path('../dataset/preprocessed/jvs_ver1/whisper10/')
RESULTS_DIR = Path('./results')


def get_common_files():
    """Find .wav files present in all three directories."""
    # Recursively find all wav files in VOICE_BASE
    # Relative path from base is used as key
    voice_files = set(
        p.relative_to(VOICE_BASE) 
        for p in VOICE_BASE.glob('**/*.wav')
    )
    
    # Check existence in other directories
    common_files = []
    for f in voice_files:
        if (PSEUDO_BASE / f).exists() and (WHISPER_BASE / f).exists():
            common_files.append(f)
            
    return sorted(list(common_files))


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


def plot_mel_spectrograms(voice_path, pseudo_path, whisper_path, output_path):
    """3つのメルスペクトログラムを描画して保存"""
    # メルスペクトログラムを抽出
    voice_mel = load_and_extract_melspec(voice_path).T
    pseudo_mel = load_and_extract_melspec(pseudo_path).T
    whisper_mel = load_and_extract_melspec(whisper_path).T
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Voice (nonpara30)を描画
    librosa.display.specshow(
        voice_mel, x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length, ax=axes[0])
    axes[0].set_ylabel('Mel Frequency (Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_title('Pseudo Whisper', fontsize=12, fontweight='bold')
    
    # Pseudo (nonpara30w)を描画
    librosa.display.specshow(
        pseudo_mel, x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length, ax=axes[1])
    axes[1].set_ylabel('Mel Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Pseudo Whisper using VITS', fontsize=12, fontweight='bold')

    # Whisper (whisper10)を描画
    img = librosa.display.specshow(
        whisper_mel, x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length, ax=axes[2])
    axes[2].set_ylabel('Mel Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Whisper', fontsize=12, fontweight='bold')

    fig.colorbar(img, ax=axes, format='%+2.0f dB')
    plt.suptitle(f'Comparison: {voice_path.name}', fontsize=16)
    
    # 保存
    plt.savefig(output_path)
    plt.close(fig) # Close to free memory


# メルスペクトログラムを描画
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    common_files = get_common_files()
    print(f"Found {len(common_files)} common files.")
    
    for f in tqdm(common_files):
        voice_p = VOICE_BASE / f
        pseudo_p = PSEUDO_BASE / f
        whisper_p = WHISPER_BASE / f
        
        # Output filename: jvs001_BASIC5000_0235.png (example)
        # Avoid slashes in filename
        safe_name = str(f).replace(os.sep, '_').replace('.wav', '.png')
        output_p = RESULTS_DIR / safe_name
        
        try:
            plot_mel_spectrograms(voice_p, pseudo_p, whisper_p, output_p)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    print(f"Done! Results saved to {RESULTS_DIR.resolve()}")
