import argparse
import csv
import numpy as np
import librosa
import pyworld as pw
from pathlib import Path
from tqdm import tqdm


# Configuration constants (inline to avoid external dependency)
n_mels = 80
n_fft = 1024
hop_length = 256


def dynamic_range_compression(
    x: np.ndarray, clip_val: float = 1e-5
) -> np.ndarray:
    """Convert to log scale with clipping."""
    return np.log(np.clip(x, clip_val, None))


def compute_mcep(wav, sr, n_mcep=24, frame_period=5.0):
    """PyWorldを使ってメルケプストラム(MCEP)を抽出する"""
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    mcep = pw.code_spectral_envelope(sp, sr, n_mcep)
    return mcep


def compute_log_mel(
    wav,
    sr,
    fft_size=n_fft,
    hop=hop_length,
    mels=n_mels
):
    """Librosaを使って対数メルスペクトログラムを抽出する"""
    # 振幅スペクトル
    D = librosa.stft(wav, n_fft=fft_size, hop_length=hop, win_length=fft_size)
    sp, _ = librosa.magphase(D)

    # メルフィルタバンク
    # ※ fmaxはサンプリングレートに合わせて調整してください (例: sr/2)
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=fft_size,
        n_mels=mels,
        fmin=0,
        fmax=None
    )
    mel_sp = np.dot(mel_basis, sp)

    # 対数変換 (dB)
    log_mel = dynamic_range_compression(mel_sp)
    return log_mel.T  # (Time, n_mels)


def calculate_mcd_with_dtw(ref_wav, gen_wav, sr, n_mcep=24):
    """DTWを用いてアライメントを取り、MCD (dB) を計算する"""
    # 1. 特徴量抽出 (MCEP)
    # n_mcepは通常24次元前後が使われます
    ref_mcep = compute_mcep(ref_wav, sr, n_mcep)
    gen_mcep = compute_mcep(gen_wav, sr, n_mcep)

    # 2. パワー項(0次元目)の除去 (音量差の影響を無視するため)
    ref_mcep_nopower = ref_mcep[:, 1:]
    gen_mcep_nopower = gen_mcep[:, 1:]

    # 3. DTWによるアライメント計算
    # librosa.sequence.dtw は (Feature, Time) の形を期待するため転置します
    d, path = librosa.sequence.dtw(
        ref_mcep_nopower.T,
        gen_mcep_nopower.T,
        metric='euclidean'
    )

    # 4. 距離の計算
    # path は (ref_index, gen_index) のタプルのリスト
    dist_sum = 0.0
    for i, j in path:
        diff = ref_mcep_nopower[i] - gen_mcep_nopower[j]
        dist_sum += np.sqrt(np.sum(diff ** 2))

    mean_dist = dist_sum / len(path)

    # 5. dB単位への変換係数
    k_mcd = (10 * np.sqrt(2)) / np.log(10)
    mcd_db = k_mcd * mean_dist

    return mcd_db


def calculate_mel_mse_with_dtw(ref_wav, gen_wav, sr):
    """DTWを用いてアライメントを取り、メルスペクトログラムのMSEを計算する"""
    # 1. 特徴量抽出 (Log Mel Spectrogram)
    ref_mel = compute_log_mel(ref_wav, sr)
    gen_mel = compute_log_mel(gen_wav, sr)

    # 2. DTWによるアライメント (メルスペクトル同士で最短経路を探索)
    d, path = librosa.sequence.dtw(ref_mel.T, gen_mel.T, metric='euclidean')

    # 3. MSE (Mean Squared Error) の計算
    sq_err_sum = 0.0
    for i, j in path:
        diff = ref_mel[i] - gen_mel[j]
        sq_err_sum += np.mean(diff ** 2)  # フレームごとのMSE

    mean_mse = sq_err_sum / len(path)

    return mean_mse


def main(args):
    ref_dir = Path(args.ref_dir)    # 正解データ (Whisper)
    gen_dir = Path(args.gen_dir)    # 生成データ (Pseudo)
    output_csv = Path(args.output_csv)

    # 検索対象のファイルリスト (生成データを基準に検索)
    gen_files = sorted(list(gen_dir.rglob("*.wav")))

    print(f"Processing {len(gen_files)} files...")

    # CSVヘッダー書き込み
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'MCD_dB', 'Mel_MSE'])

        for gen_path in tqdm(gen_files):
            # 相対パスを使って対応する正解ファイルを探す
            rel_path = gen_path.relative_to(gen_dir)
            ref_path = ref_dir / rel_path

            # 正解データが見つからない場合はスキップ
            if not ref_path.exists():
                # ファイル名だけで探してみる救済措置
                candidates = list(ref_dir.rglob(gen_path.name))
                if candidates:
                    ref_path = candidates[0]
                else:
                    # print(f"Ref not found: {gen_path.name}")
                    continue

            try:
                # 音声読み込み
                ref_wav, _ = librosa.load(ref_path, sr=args.sr)
                gen_wav, _ = librosa.load(gen_path, sr=args.sr)

                # 指標計算
                mcd = calculate_mcd_with_dtw(ref_wav, gen_wav, args.sr)
                mse = calculate_mel_mse_with_dtw(ref_wav, gen_wav, args.sr)

                # CSVに書き込み
                writer.writerow([gen_path.name, f"{mcd:.4f}", f"{mse:.4f}"])

            except Exception as e:
                print(f"Error processing {gen_path.name}: {e}")

    print(f"Done! Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate MCD and Mel-MSE with DTW")
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="Reference directory (Real Whisper)"
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Generated directory (Pseudo Whisper)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/metrics.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sampling rate"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    main(args)
