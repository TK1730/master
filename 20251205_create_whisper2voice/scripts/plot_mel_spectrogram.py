"""Plot mel spectrograms for comparison."""
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path


def plot_mel_spectrogram(
    wav_path: str,
    title: str,
    output_path: str,
    sr: int = 22050,
    duration: float = 2.4
) -> None:
    """Plot mel spectrogram from audio file.

    Args:
        wav_path: Path to wav file
        title: Title for the plot
        output_path: Path to save the figure
        sr: Sampling rate
        duration: Duration in seconds to trim
    """
    # Load audio
    y, _ = librosa.load(wav_path, sr=sr)

    # Trim to duration
    max_samples = int(sr * duration)
    y = y[:max_samples]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=80, n_fft=1024, hop_length=256
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec_db,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length=256,
        ax=ax
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Main function to plot mel spectrograms."""
    # Sample file
    sample_file = "BASIC5000_0235.wav"

    # Paths
    ref_path = f"dataset/nonpara30/jvs001/wav/{sample_file}"
    gen1_path = f"results/generated/whisper_converted_v2/jvs001/wav/{sample_file}"
    gen2_path = f"results/generated/whisper10/jvs001/wav/{sample_file}"

    # Output directory
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot each
    plot_mel_spectrogram(
        ref_path,
        "Ground Truth (nonpara30)",
        output_dir / "mel_nonpara30.png"
    )
    plot_mel_spectrogram(
        gen1_path,
        "whisper_converted_v2",
        output_dir / "mel_whisper_converted_v2.png"
    )
    plot_mel_spectrogram(
        gen2_path,
        "whisper10",
        output_dir / "mel_whisper10.png"
    )


if __name__ == "__main__":
    main()
