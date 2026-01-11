from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import config, functions

# Base Directories
# Target: DATASET/preprocessed/jvs_ver1/nonpara30
# Note: The README says 'dataset/nonpara30' but often these are symlinked or we need to find the real path.
# Assuming standard structure relative to script: ../dataset/nonpara30
# Based on previous listing, 'dataset/nonpara30' exists in the project root.
TARGET_BASE = Path('../dataset/nonpara30') 

# Converted V2: ../results/generated/whisper_converted_v2
PSEUDO_CONV_BASE = Path('../results/generated/whisper_converted_v2')

# Converted Whisper10: ../results/generated/whisper10
WHISPER_CONV_BASE = Path('../results/generated/whisper10')

# Output Directory
RESULTS_DIR = Path('../results/msp_images')


def get_common_files():
    """Find .wav files present in all three directories."""
    # Find all wav files in PSEUDO_CONV_BASE
    # These are the generated files we want to evaluate.
    # Structure in generated: jvsXXX/wav/filename.wav (likely)
    
    pseudo_files = set(
        p.relative_to(PSEUDO_CONV_BASE) 
        for p in PSEUDO_CONV_BASE.glob('**/*.wav')
    )
    
    # Check existence in other directories
    common_files = []
    for f in pseudo_files:
        if (TARGET_BASE / f).exists() and (WHISPER_CONV_BASE / f).exists():
            common_files.append(f)
            
    return sorted(list(common_files))


def load_and_extract_melspec(audio_path, sr=22050):
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


def plot_mel_spectrograms(target_path, pseudo_conv_path, whisper_conv_path, output_path):
    """3つのメルスペクトログラムを描画して保存"""
    # メルスペクトログラムを抽出
    target_mel = load_and_extract_melspec(target_path).T
    pseudo_conv_mel = load_and_extract_melspec(pseudo_conv_path).T
    whisper_conv_mel = load_and_extract_melspec(whisper_conv_path).T
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Target (Ground Truth)
    librosa.display.specshow(
        target_mel, x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length, ax=axes[0])
    axes[0].set_ylabel('Mel Frequency (Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_title('Target (nonpara30)', fontsize=12, fontweight='bold')
    
    # Pseudo Conv (from whisper_converted_v2)
    librosa.display.specshow(
        pseudo_conv_mel, x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length, ax=axes[1])
    axes[1].set_ylabel('Mel Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Converted (from Pseudo Whisper)', fontsize=12, fontweight='bold')

    # Whisper Conv (from whisper10)
    img = librosa.display.specshow(
        whisper_conv_mel, x_axis='time', y_axis='mel',
        sr=22050, hop_length=config.hop_length, ax=axes[2])
    axes[2].set_ylabel('Mel Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Converted (from Whisper10)', fontsize=12, fontweight='bold')

    fig.colorbar(img, ax=axes, format='%+2.0f dB')
    plt.suptitle(f'Comparison: {target_path.name}', fontsize=16)
    
    # 保存
    plt.savefig(output_path)
    plt.close(fig) # Close to free memory


if __name__ == "__main__":
    # Change to script directory to ensure relative paths work if run from project root
    os.chdir(Path(__file__).parent)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Searching for common files...")
    common_files = get_common_files()
    print(f"Found {len(common_files)} common files.")
    
    if len(common_files) == 0:
        print("Debug Info:")
        print(f"TARGET_BASE: {TARGET_BASE.resolve()} (Exists: {TARGET_BASE.exists()})")
        print(f"PSEUDO_CONV_BASE: {PSEUDO_CONV_BASE.resolve()} (Exists: {PSEUDO_CONV_BASE.exists()})")
        print(f"WHISPER_CONV_BASE: {WHISPER_CONV_BASE.resolve()} (Exists: {WHISPER_CONV_BASE.exists()})")
    
    for f in tqdm(common_files):
        target_p = TARGET_BASE / f
        pseudo_conv_p = PSEUDO_CONV_BASE / f
        whisper_conv_p = WHISPER_CONV_BASE / f
        
        # Output filename: jvs001_BASIC5000_0235.png
        safe_name = str(f).replace(os.sep, '_').replace('.wav', '.png')
        output_p = RESULTS_DIR / safe_name
        
        try:
            plot_mel_spectrograms(target_p, pseudo_conv_p, whisper_conv_p, output_p)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    print(f"Done! Results saved to {RESULTS_DIR.resolve()}")
