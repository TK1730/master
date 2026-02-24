"""
Mel-Spectrogram Visualization Module

This module provides functionality to compute and visualize mel-spectrograms
from audio files in different datasets (whisper10, nonpara30w_ver2).
"""
import matplotlib
matplotlib.use('Agg')  # noqa: E402

import numpy as np  # noqa: E402
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, List, Tuple  # noqa: E402


# Configuration constants (matching analyze_features.py for consistency)
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
F_MIN = 0
F_MAX = None  # 8kHz upper limit for better visualization


def dynamic_range_compression(
    x: np.ndarray, clip_val: float = 1e-5
) -> np.ndarray:
    """
    Convert to log scale with clipping.

    Args:
        x: Input array (mel spectrogram)
        clip_val: Minimum value for clipping

    Returns:
        Log-scaled array
    """
    return np.log(np.clip(x, clip_val, None))


def compute_mel_spectrogram(
    wav: np.ndarray,
    sr: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: Optional[float] = F_MAX
) -> np.ndarray:
    """
    Compute mel-spectrogram from audio waveform.

    Args:
        wav: Audio waveform
        sr: Sampling rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        Mel-spectrogram in dB scale (time, n_mels)
    """
    # Compute STFT
    D = librosa.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft
    )
    sp, _ = librosa.magphase(D)

    # Create mel filter bank
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )

    # Apply mel filter bank
    mel_sp = np.dot(mel_basis, sp)

    # Convert to dB scale
    log_mel = dynamic_range_compression(mel_sp)

    return log_mel.T  # Return as (time, n_mels)


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot mel-spectrogram with proper axes and labels.

    Args:
        mel_spec: Mel-spectrogram (time, n_mels)
        sr: Sampling rate
        hop_length: Hop length used for STFT
        title: Plot title
        output_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Create mel spectrogram display (use default colormap)
    img = librosa.display.specshow(
        mel_spec.T,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        fmin=F_MIN,
        fmax=F_MAX
    )

    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_single_file(
    audio_path: str,
    output_dir: str = "results/mel_spectrograms",
    sr: int = 22050
) -> None:
    """
    Visualize mel-spectrogram from a single audio file.

    Args:
        audio_path: Path to the audio file
        output_dir: Output directory for saved plot
        sr: Sampling rate for audio loading
    """
    audio_file = Path(audio_path)

    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing single file: {audio_file.name}")
    print(f"{'='*60}")

    try:
        # Load audio
        wav, _ = librosa.load(audio_file, sr=sr)
        print(f"Duration: {len(wav)/sr:.2f}s")

        # Compute mel-spectrogram
        mel_spec = compute_mel_spectrogram(wav, sr)
        print(f"Mel-spec shape: {mel_spec.shape}")

        # Create plot title
        plot_title = f"Mel-Spectrogram - {audio_file.name}"

        # Save path - include parent directories to make unique
        # Extract dataset name from path
        parts = audio_file.parts
        if 'dataset' in parts:
            dataset_idx = parts.index('dataset')
            if dataset_idx + 1 < len(parts):
                dataset_name = parts[dataset_idx + 1]
                safe_name = f"{dataset_name}_{audio_file.stem}"
            else:
                safe_name = audio_file.stem
        else:
            safe_name = audio_file.stem

        save_path = output_path / f"{safe_name}_melspec.png"

        # Plot and save
        plot_mel_spectrogram(
            mel_spec,
            sr,
            HOP_LENGTH,
            plot_title,
            save_path
        )

        print(f"\n{'='*60}")
        print(f"Visualization saved to: {save_path.absolute()}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")


def visualize_dataset_samples(
    dataset_paths: List[str],
    n_samples: int = 3,
    output_dir: str = "results/mel_spectrograms",
    sr: int = 22050
) -> None:
    """
    Visualize mel-spectrograms from multiple datasets.

    Args:
        dataset_paths: List of dataset directory paths
        n_samples: Number of samples to visualize per dataset
        output_dir: Output directory for saved plots
        sr: Sampling rate for audio loading
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_path_str in dataset_paths:
        dataset_path = Path(dataset_path_str)

        if not dataset_path.exists():
            print(f"Warning: Dataset path not found: {dataset_path}")
            continue

        # Get dataset name
        dataset_name = dataset_path.name
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Find all wav files
        wav_files = sorted(list(dataset_path.rglob("*.wav")))

        if not wav_files:
            print(f"No wav files found in {dataset_path}")
            continue

        print(f"Found {len(wav_files)} audio files")

        # Select samples (evenly distributed)
        if len(wav_files) > n_samples:
            step = len(wav_files) // n_samples
            sample_files = [
                wav_files[i * step] for i in range(n_samples)
            ]
        else:
            sample_files = wav_files

        # Process each sample
        for idx, wav_file in enumerate(sample_files, 1):
            try:
                file_info = f"[{idx}/{len(sample_files)}]"
                print(f"\n{file_info} Processing: {wav_file.name}")

                # Load audio
                wav, _ = librosa.load(wav_file, sr=sr)
                print(f"  Duration: {len(wav)/sr:.2f}s")

                # Compute mel-spectrogram
                mel_spec = compute_mel_spectrogram(wav, sr, f_max=None)
                print(f"  Mel-spec shape: {mel_spec.shape}")

                # Create plot title
                relative_path = wav_file.relative_to(dataset_path)
                plot_title = f"{dataset_name} - {relative_path}"

                # Save path
                safe_name = str(relative_path)
                safe_name = safe_name.replace('\\', '_')
                safe_name = safe_name.replace('/', '_')
                file_name = f"{dataset_name}_{safe_name}.png"
                save_path = output_path / file_name

                # Plot and save
                plot_mel_spectrogram(
                    mel_spec,
                    sr,
                    HOP_LENGTH,
                    plot_title,
                    save_path
                )

            except Exception as e:
                print(f"  Error processing {wav_file.name}: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_path.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Process BASIC5000_0235.wav from both datasets
    files_to_process = [
        "dataset/nonpara30w_ver2/jvs001/wav/BASIC5000_0235.wav",
        "dataset/whisper10/jvs001/wav/BASIC5000_0235.wav"
    ]

    for audio_path in files_to_process:
        visualize_single_file(
            audio_path=audio_path,
            output_dir="results/mel_spectrograms",
            sr=22050
        )
