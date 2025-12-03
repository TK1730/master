import argparse
import os
import sys
from pathlib import Path
import numpy as np
import librosa
import pyworld
import pysptk
from scipy.spatial.distance import euclidean
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parents[2]))
import utils.config as config
import utils.functions as functions


def get_mcep(wav, sr, n_fft=config.n_fft, n_mels=config.n_mels, dim=24):
    """
    Extract Mel-Cepstral Coefficients (MCEPs) using pyworld and pysptk.
    """
    # Convert to float64 for pyworld
    wav = wav.astype(np.float64)

    # F0 estimation
    _f0, t = pyworld.dio(wav, sr, frame_period=config.hop_length / sr * 1000)
    f0 = pyworld.stonemask(wav, _f0, t, sr)

    # Spectral Envelope
    sp = pyworld.cheaptrick(wav, f0, t, sr)

    # Convert to Mel-Cepstrum
    alpha = pysptk.util.mcepalpha(sr)
    mcep = pysptk.sp2mc(sp, dim, alpha)

    return mcep

def get_melspec(wav, sr):
    """
    Extract Mel-Spectrogram using utils.functions.wav2msp logic.
    """
    D = librosa.stft(y=wav, n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.win_length, pad_mode='reflect').T
    sp, phase = librosa.magphase(D)
    msp = np.matmul(sp, config.mel_filter)
    log_msp = np.log10(np.maximum(msp, 1e-10))
    return log_msp

def calculate_mcd(ref_mcep, gen_mcep):
    """
    Calculate Mel-Cepstral Distortion (MCD) using DTW.
    """
    # librosa.sequence.dtw expects (n_features, n_frames)
    # mcep is (n_frames, dim) -> transpose to (dim, n_frames)
    D, wp = librosa.sequence.dtw(ref_mcep.T, gen_mcep.T, metric='euclidean')
    
    # wp is list of (ref_idx, gen_idx) in reverse order
    path = wp[::-1]
    
    ref_aligned = ref_mcep[path[:, 0]]
    gen_aligned = gen_mcep[path[:, 1]]
    
    diff = ref_aligned - gen_aligned
    # Sum of squared differences per frame
    dist_sq = np.sum(diff**2, axis=1)
    # Root of sum of squared differences
    dist_rmse = np.sqrt(dist_sq)
    
    # Mean over frames
    mean_dist = np.mean(dist_rmse)
    
    # Scale
    mcd = (10 * np.sqrt(2) / np.log(10)) * mean_dist
    
    return mcd

def calculate_mse(ref_mel, gen_mel):
    """
    Calculate Mean Squared Error (MSE) on Mel-Spectrogram using DTW.
    """
    # librosa.sequence.dtw expects (n_features, n_frames)
    D, wp = librosa.sequence.dtw(ref_mel.T, gen_mel.T, metric='euclidean')
    
    path = wp[::-1]
    
    ref_aligned = ref_mel[path[:, 0]]
    gen_aligned = gen_mel[path[:, 1]]
    
    # MSE
    mse = np.mean((ref_aligned - gen_aligned)**2)
    
    return mse

def main():
    parser = argparse.ArgumentParser(description="Calculate MCD and MSE for phonation mode comparison.")
    parser.add_argument('--gen_dir', type=str, required=True, help="Directory containing VITS generated audio")
    parser.add_argument('--target_dir', type=str, required=True, help="Directory containing target pseudo-whisper audio")
    parser.add_argument('--ref_dir', type=str, required=True, help="Directory containing reference whisper audio")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # Recursive search for wav files
    gen_files = sorted(list(Path(args.gen_dir).rglob('*.wav')))

    print(f"Found {len(gen_files)} files in {args.gen_dir}")

    for gen_path in tqdm(gen_files):
        # Get relative path to maintain structure (e.g., jvs001/VOICEACTRESS100_001.wav)
        rel_path = gen_path.relative_to(args.gen_dir)

        target_path = Path(args.target_dir) / rel_path
        ref_path = Path(args.ref_dir) / rel_path

        if not target_path.exists():
            # Try checking if filename matches but directory structure is different?
            # For now assume same structure.
            # print(f"Warning: Target file not found for {rel_path}")
            continue
        if not ref_path.exists():
            # print(f"Warning: Ref file not found for {rel_path}")
            continue

        try:
            # Load audio
            gen_wav, _ = librosa.load(gen_path, sr=config.sr)
            target_wav, _ = librosa.load(target_path, sr=config.sr)
            ref_wav, _ = librosa.load(ref_path, sr=config.sr)

            # Normalize Gen audio
            gen_wav = functions.loudness_normalize(gen_wav, config.lufs_mix, load=False)

            # Extract Features
            # MCEP
            gen_mcep = get_mcep(gen_wav, config.sr)
            target_mcep = get_mcep(target_wav, config.sr)
            ref_mcep = get_mcep(ref_wav, config.sr)

            # MelSpec
            gen_mel = get_melspec(gen_wav, config.sr)
            target_mel = get_melspec(target_wav, config.sr)
            ref_mel = get_melspec(ref_wav, config.sr)

            # Calculate Metrics
            # 1. Gen vs Target
            mcd_gen_target = calculate_mcd(target_mcep, gen_mcep)
            mse_gen_target = calculate_mse(target_mel, gen_mel)

            # 2. Ref vs Target (Baseline)
            # Or Gen vs Ref?
            # User request: "Referencing pseudo-whisper (Target), calculate ... for VITS (Gen) and Whisper (Ref)"
            # So we compare Gen to Target, and Ref to Target.
            mcd_ref_target = calculate_mcd(target_mcep, ref_mcep)
            mse_ref_target = calculate_mse(target_mel, ref_mel)
            
            results.append({
                'filename': str(rel_path),
                'mcd_gen_target': mcd_gen_target,
                'mse_gen_target': mse_gen_target,
                'mcd_ref_target': mcd_ref_target,
                'mse_ref_target': mse_ref_target
            })
            
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    csv_path = Path(args.output_dir) / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    
    # Calculate averages
    print("\nAverage Results:")
    print(df.mean(numeric_only=True))
    
    summary_path = Path(args.output_dir) / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(str(df.mean(numeric_only=True)))

if __name__ == '__main__':
    main()
