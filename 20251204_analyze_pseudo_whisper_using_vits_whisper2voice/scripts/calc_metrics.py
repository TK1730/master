import argparse
import csv
import numpy as np
import librosa
import pyworld as pw
from pathlib import Path
from tqdm import tqdm


# Constants (from utils/config.py)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 0
FMAX = 8000  # or None (sampling_rate/2)


def compute_mcep(wav, sr, n_mcep=24, frame_period=5.0):
    """
    Compute Mel-Cepstral coefficients (MCEP) using PyWorld.
    
    Args:
        wav: Audio waveform (numpy array)
        sr: Sampling rate
        n_mcep: Number of MCEP coefficients
        frame_period: Frame period in milliseconds
    
    Returns:
        mcep: Mel-cepstral coefficients (frames x n_mcep)
    """
    wav = wav.astype(np.float64)
    # F0 estimation using DIO
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    mcep = pw.code_spectral_envelope(sp, sr, n_mcep)
    return mcep


def compute_modulation_spectrum(wav, sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 n_mels=N_MELS):
    """
    Compute modulation spectrum (MSP).
    
    1. Calculate log mel-spectrogram
    2. Apply FFT along time axis for each mel band
    3. Take amplitude to get modulation spectrum
    4. Average across all mel bands to get global modulation spectrum
    
    Args:
        wav: Audio waveform
        sr: Sampling rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bands
    
    Returns:
        msp_avg: Average modulation spectrum (modulation_frequency,)
    """
    # 1. Log mel-spectrogram
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length,
                     win_length=n_fft)
    sp, _ = librosa.magphase(D)
    
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=FMIN,
        fmax=FMAX
    )
    mel_sp = np.dot(mel_basis, sp)
    log_mel = np.log(mel_sp + 1e-6)  # (n_mels, Time)

    # 2. FFT along time axis
    # Remove DC component (static energy) by subtracting mean
    log_mel_centered = log_mel - np.mean(log_mel, axis=1, keepdims=True)
    
    # FFT along time axis
    msp = np.abs(np.fft.rfft(log_mel_centered, axis=1))
    
    # Normalize by length
    msp = msp / log_mel.shape[1]
    
    # 3. Average across all mel bands
    # Result is (modulation_frequency,)
    msp_avg = np.mean(msp, axis=0)
    
    return msp_avg


def calculate_mcd_with_dtw(ref_wav, gen_wav, sr, n_mcep=24):
    """
    Calculate MCD (Mel-Cepstral Distortion) using DTW alignment.
    
    Args:
        ref_wav: Reference audio waveform
        gen_wav: Generated audio waveform
        sr: Sampling rate
        n_mcep: Number of MCEP coefficients
    
    Returns:
        mcd_db: MCD value in dB
    """
    ref_mcep = compute_mcep(ref_wav, sr, n_mcep)
    gen_mcep = compute_mcep(gen_wav, sr, n_mcep)

    # Remove power coefficient (0th dimension)
    ref_mcep_nopower = ref_mcep[:, 1:]
    gen_mcep_nopower = gen_mcep[:, 1:]

    # DTW alignment
    # librosa.sequence.dtw expects (Feature, Time)
    d, path = librosa.sequence.dtw(
        ref_mcep_nopower.T,
        gen_mcep_nopower.T,
        metric='euclidean'
    )

    # Distance calculation
    dist_sum = 0.0
    for i, j in path:
        diff = ref_mcep_nopower[i] - gen_mcep_nopower[j]
        dist_sum += np.sqrt(np.sum(diff ** 2))

    mean_dist = dist_sum / len(path)

    # Convert to dB
    # k = 10 * sqrt(2) / ln(10) approx 6.14
    k_mcd = (10 * np.sqrt(2)) / np.log(10)
    mcd_db = k_mcd * mean_dist

    return mcd_db


def calculate_msp_distance(ref_wav, gen_wav, sr):
    """
    Calculate Euclidean distance between modulation spectra.
    MSP is a global statistic (time average), so DTW is not required.
    
    Args:
        ref_wav: Reference audio waveform
        gen_wav: Generated audio waveform
        sr: Sampling rate
    
    Returns:
        dist: Euclidean distance between modulation spectra
    """
    ref_msp = compute_modulation_spectrum(ref_wav, sr)
    gen_msp = compute_modulation_spectrum(gen_wav, sr)
    
    # Align lengths by interpolation for comparison
    target_len = 100  # Arbitrary resolution for comparison
    
    x_ref = np.linspace(0, 1, len(ref_msp))
    x_gen = np.linspace(0, 1, len(gen_msp))
    x_target = np.linspace(0, 1, target_len)
    
    ref_msp_interp = np.interp(x_target, x_ref, ref_msp)
    gen_msp_interp = np.interp(x_target, x_gen, gen_msp)
    
    # Euclidean distance
    dist = np.sqrt(np.sum((ref_msp_interp - gen_msp_interp) ** 2))
    
    return dist


def main(args):
    """
    Main function to calculate acoustic metrics.
    """
    target_dir = Path(args.target_dir)  # Target (voiced speech)
    vits_gen_dir = Path(args.vits_gen_dir)  # VITS pseudo-whisper conversion
    whisper_conv_dir = Path(args.whisper_conv_dir)  # Whisper conversion
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "metrics.csv"

    # Find common files
    vits_files = sorted(list(vits_gen_dir.rglob("*.wav")))
    
    print(f"Number of VITS generated files: {len(vits_files)}")
    
    results = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename',
            'MCD_VITSGen_vs_Target',
            'MSP_Dist_VITSGen_vs_Target',
            'MCD_WhisperConv_vs_Target',
            'MSP_Dist_WhisperConv_vs_Target'
        ])
        
        for vits_path in tqdm(vits_files):
            rel_path = vits_path.relative_to(vits_gen_dir)
            target_path = target_dir / rel_path
            whisper_conv_path = whisper_conv_dir / rel_path
            
            # Check if corresponding files exist
            if not target_path.exists():
                # Try searching by name
                candidates = list(target_dir.rglob(vits_path.name))
                if candidates:
                    target_path = candidates[0]
                else:
                    continue
            
            if not whisper_conv_path.exists():
                candidates = list(whisper_conv_dir.rglob(vits_path.name))
                if candidates:
                    whisper_conv_path = candidates[0]
                else:
                    continue
            
            try:
                # Load audio
                vits_wav, _ = librosa.load(vits_path, sr=args.sr)
                target_wav, _ = librosa.load(target_path, sr=args.sr)
                whisper_conv_wav, _ = librosa.load(whisper_conv_path,
                                                    sr=args.sr)
                
                # Calculate metrics: VITS Gen vs Target
                mcd_vits = calculate_mcd_with_dtw(target_wav, vits_wav,
                                                   args.sr)
                msp_dist_vits = calculate_msp_distance(target_wav, vits_wav,
                                                        args.sr)
                
                # Calculate metrics: Whisper Conv vs Target (baseline)
                mcd_whisper = calculate_mcd_with_dtw(target_wav,
                                                      whisper_conv_wav,
                                                      args.sr)
                msp_dist_whisper = calculate_msp_distance(target_wav,
                                                           whisper_conv_wav,
                                                           args.sr)
                
                writer.writerow([
                    vits_path.name,
                    f"{mcd_vits:.4f}", f"{msp_dist_vits:.4f}",
                    f"{mcd_whisper:.4f}", f"{msp_dist_whisper:.4f}"
                ])
                
                results.append({
                    'mcd_vits': mcd_vits,
                    'msp_vits': msp_dist_vits,
                    'mcd_whisper': mcd_whisper,
                    'msp_whisper': msp_dist_whisper
                })
                
            except Exception as e:
                print(f"Error processing {vits_path.name}: {e}")

    # Summary statistics
    if results:
        avg_mcd_vits = np.mean([r['mcd_vits'] for r in results])
        std_mcd_vits = np.std([r['mcd_vits'] for r in results])
        
        avg_msp_vits = np.mean([r['msp_vits'] for r in results])
        std_msp_vits = np.std([r['msp_vits'] for r in results])
        
        avg_mcd_whisper = np.mean([r['mcd_whisper'] for r in results])
        std_mcd_whisper = np.std([r['mcd_whisper'] for r in results])
        
        avg_msp_whisper = np.mean([r['msp_whisper'] for r in results])
        std_msp_whisper = np.std([r['msp_whisper'] for r in results])
        
        print("\n" + "="*80)
        print("Summary Results (集計結果)")
        print("="*80)
        print(f"{'Metric':<20} | {'Target vs Whisper Conv':<30} | "
              f"{'Target vs VITS Gen':<30}")
        print("-" * 80)
        print(f"{'MCD (dB)':<20} | {avg_mcd_whisper:.4f} ± "
              f"{std_mcd_whisper:.4f}{'':<12} | {avg_mcd_vits:.4f} ± "
              f"{std_mcd_vits:.4f}")
        print(f"{'MSP (Distance)':<20} | {avg_msp_whisper:.4f} ± "
              f"{std_msp_whisper:.4f}{'':<12} | {avg_msp_vits:.4f} ± "
              f"{std_msp_vits:.4f}")
        print("="*80)
        print(f"Results saved to {output_csv}")
        
        # Save summary to separate file
        summary_file = output_dir / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Summary Results (集計結果)\n")
            f.write("="*80 + "\n")
            f.write(f"{'Metric':<20} | {'Target vs Whisper Conv':<30} | "
                    f"{'Target vs VITS Gen':<30}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'MCD (dB)':<20} | {avg_mcd_whisper:.4f} ± "
                    f"{std_mcd_whisper:.4f}{'':<12} | {avg_mcd_vits:.4f} ± "
                    f"{std_mcd_vits:.4f}\n")
            f.write(f"{'MSP (Distance)':<20} | {avg_msp_whisper:.4f} ± "
                    f"{std_msp_whisper:.4f}{'':<12} | {avg_msp_vits:.4f} ± "
                    f"{std_msp_vits:.4f}\n")
            f.write("="*80 + "\n")
        print(f"Summary saved to {summary_file}")
    else:
        print("No valid results obtained. Please check paths and data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acoustic analysis: Calculate MSP and MCD metrics"
    )
    parser.add_argument(
        "--vits_gen_dir",
        type=str,
        required=True,
        help="Directory of VITS-generated audio"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory of target audio (voiced speech)"
    )
    parser.add_argument(
        "--whisper_conv_dir",
        type=str,
        required=True,
        help="Directory of whisper conversion audio (baseline)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sampling rate"
    )
    
    args = parser.parse_args()
    main(args)
