import argparse
import csv
import numpy as np
import librosa
import pyworld as pw
from pathlib import Path
from tqdm import tqdm

import utils.functions as functions


def compute_mcep(wav, sr, n_mcep=24, frame_period=5.0):
    """PyWorldを使ってメルケプストラム(MCEP)を抽出する"""
    wav = wav.astype(np.float64)
    # ささやき声の分析には harvest を推奨
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    sp = pw.cheaptrick(wav, f0, t, sr)
    mcep = pw.code_spectral_envelope(sp, sr, n_mcep)
    return mcep


def compute_log_mel(
    wav,
    sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
):
    """Librosaを使って対数メルスペクトログラムを抽出する (Power-to-dB)"""
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    sp, _ = librosa.magphase(D)

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=None
    )
    mel_sp = np.dot(mel_basis, sp)
    log_mel = functions.dynamic_range_compression(mel_sp)
    return log_mel.T


def calculate_metrics(ref_wav, gen_wav, sr):
    """DTWを用いてアライメントを取り、MCDとMSEを計算する"""
    # --- MCD (Mel-Cepstral Distortion) ---
    ref_mcep = compute_mcep(ref_wav, sr)
    gen_mcep = compute_mcep(gen_wav, sr)

    # 0次元目(パワー)を除去
    ref_mcep_nopower = ref_mcep[:, 1:]
    gen_mcep_nopower = gen_mcep[:, 1:]

    # DTW (MCEP)
    d, path = librosa.sequence.dtw(
        ref_mcep_nopower.T, gen_mcep_nopower.T, metric='euclidean')

    # 距離計算
    dist_sum = 0.0
    for i, j in path:
        diff = ref_mcep_nopower[i] - gen_mcep_nopower[j]
        dist_sum += np.sqrt(np.sum(diff ** 2))
    mean_dist = dist_sum / len(path)

    # dB変換 (10*sqrt(2)/ln(10) ≈ 6.14)
    mcd_db = (10 * np.sqrt(2) / np.log(10)) * mean_dist

    # --- Mel-MSE (Mean Squared Error) ---
    ref_mel = compute_log_mel(ref_wav, sr)
    gen_mel = compute_log_mel(gen_wav, sr)

    # DTW (Mel) - スペクトル形状で再度アライメントを取るのが正確
    d_mel, path_mel = librosa.sequence.dtw(
        ref_mel.T, gen_mel.T, metric='euclidean')

    sq_err_sum = 0.0
    for i, j in path_mel:
        diff = ref_mel[i] - gen_mel[j]
        sq_err_sum += np.mean(diff ** 2)
    mean_mse = sq_err_sum / len(path_mel)

    return mcd_db, mean_mse


def main(args):
    ref_dir = Path(args.ref_dir)        # 正解: whisper10
    topline_dir = Path(args.gen_dir)    # 比較対象: whisper10_resynth
    filter_dir = Path(args.filter_dir)  # フィルタ条件: nonpara30w_ver1
    output_csv = Path(args.output_csv)

    # 1. フィルタリング用リストの作成
    # filter_dir (ver1) に存在するファイル名だけを取得
    target_filenames = set()
    for f in filter_dir.rglob("*.wav"):
        target_filenames.add(f.name)

    print(
        f"Filter target files: {len(target_filenames)}"
        f" (from {filter_dir.name})"
    )

    results = []

    # 2. Toplineデータの走査と計算
    # Toplineフォルダにあるファイルのうち、ターゲットに含まれるものだけ処理
    files_processed = 0

    # CSV書き込み準備
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'MCD_dB', 'Mel_MSE'])

        # プログレスバー用にリスト化してソート
        all_topline_files = sorted(list(topline_dir.rglob("*.wav")))

        for gen_path in tqdm(all_topline_files):
            # ファイル名チェック
            if gen_path.name not in target_filenames:
                continue

            # 対応する正解ファイル(ref)を探す
            # 相対パスで一致するか確認
            rel_path = gen_path.relative_to(topline_dir)
            ref_path = ref_dir / rel_path

            if not ref_path.exists():
                # 見つからない場合はスキップ
                continue

            try:
                # 音声読み込み
                ref_wav, _ = librosa.load(ref_path, sr=args.sr)
                gen_wav, _ = librosa.load(gen_path, sr=args.sr)

                # 指標計算
                mcd, mse = calculate_metrics(ref_wav, gen_wav, args.sr)

                # 結果保存
                writer.writerow([gen_path.name, f"{mcd:.4f}", f"{mse:.4f}"])
                results.append((mcd, mse))
                files_processed += 1

            except Exception as e:
                print(f"Error processing {gen_path.name}: {e}")

    # 3. 全体平均の表示
    if files_processed > 0:
        avg_mcd = np.mean([r[0] for r in results])
        avg_mse = np.mean([r[1] for r in results])
        print(f"\n[Done] Processed {files_processed} files.")
        print(f"Average MCD: {avg_mcd:.4f} dB")
        print(f"Average MSE: {avg_mse:.4f}")
    else:
        print("\n[Warning] No matching files found to process.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Topline Metrics with Filtering")
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
        help="Topline directory (Resynthesized Whisper)"
    )
    parser.add_argument(
        "--filter_dir",
        type=str,
        required=True,
        help="Directory to filter filenames (e.g., ver1)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/metrics_topline.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sampling rate"
    )

    args = parser.parse_args()
    main(args)
