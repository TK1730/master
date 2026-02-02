import argparse
import csv
import numpy as np
import librosa
import pyworld as pw
from pathlib import Path
from tqdm import tqdm

from utils import functions
from utils import config


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
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    n_mels=config.n_mels
):
    """Librosaを使って対数メルスペクトログラムを抽出する"""
    # 振幅スペクトル
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    sp, _ = librosa.magphase(D)

    # メルフィルタバンク
    # ※ fmaxはサンプリングレートに合わせて調整してください (例: sr/2)
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=None
    )
    mel_sp = np.dot(mel_basis, sp)

    # 対数変換 (dB)
    log_mel = functions.dynamic_range_compression(mel_sp)
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
    """DTWを用いてアライメントを取り、メルスペクトログラムのMSEを計算する

    Returns:
        tuple: (mean_mse, path) - MSEとアライメントパス
    """
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

    return mean_mse, path


def extract_world_features(wav, sr, hop_length=None):
    """PyWorldを使ってF0, SP, APを抽出する

    Args:
        wav: 音声波形データ
        sr: サンプリングレート
        hop_length: ホップ長（サンプル数）。Noneの場合はデフォルト値を使用

    Returns:
        tuple: (f0, sp, ap) - 基本周波数、スペクトル包絡、非周期性指標
    """
    wav = wav.astype(np.float64)

    # hop_lengthからframe_period (ms) を計算
    # メルスペクトログラムと同じフレーム周期にする
    if hop_length is None:
        hop_length = config.hop_length
    frame_period = (hop_length / sr) * 1000.0

    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    ap = pw.d4c(wav, f0, t, sr)

    return f0, sp, ap


def calculate_f0_mse_with_alignment(ref_f0, gen_f0, path):
    """アライメントパスを使ってF0のMSEを計算する

    Args:
        ref_f0: リファレンス音声のF0
        gen_f0: 生成音声のF0
        path: DTWアライメントパス

    Returns:
        float: F0のMSE (有声区間のみ)
    """
    sq_err_sum = 0.0
    valid_count = 0

    for i, j in path:
        # インデックスチェック
        if i >= len(ref_f0) or j >= len(gen_f0):
            continue

        # 有声区間のみ計算 (F0 > 0)
        if ref_f0[i] > 0 and gen_f0[j] > 0:
            sq_err_sum += (ref_f0[i] - gen_f0[j]) ** 2
            valid_count += 1

    if valid_count == 0:
        return 0.0

    return sq_err_sum / valid_count


def calculate_log_f0_rmse_with_alignment(ref_f0, gen_f0, path):
    """アライメントパスを使ってログスケールF0のRMSEを計算する

    Args:
        ref_f0: リファレンス音声のF0
        gen_f0: 生成音声のF0
        path: DTWアライメントパス

    Returns:
        float: ログF0のRMSE (有声区間のみ)
    """
    sq_err_sum = 0.0
    valid_count = 0

    for i, j in path:
        # インデックスチェック
        if i >= len(ref_f0) or j >= len(gen_f0):
            continue

        # 有声区間のみ計算 (F0 > 0)
        if ref_f0[i] > 0 and gen_f0[j] > 0:
            log_ref = np.log(ref_f0[i])
            log_gen = np.log(gen_f0[j])
            sq_err_sum += (log_ref - log_gen) ** 2
            valid_count += 1

    if valid_count == 0:
        return 0.0

    return np.sqrt(sq_err_sum / valid_count)


def calculate_sp_mse_with_alignment(ref_sp, gen_sp, path):
    """アライメントパスを使ってスペクトル包絡(SP)のMSEを計算する

    Args:
        ref_sp: リファレンス音声のスペクトル包絡
        gen_sp: 生成音声のスペクトル包絡
        path: DTWアライメントパス

    Returns:
        float: SPのMSE
    """
    sq_err_sum = 0.0
    valid_count = 0

    for i, j in path:
        # インデックスチェック
        if i >= len(ref_sp) or j >= len(gen_sp):
            continue

        # NaNやInfをスキップ（念のため）
        if not (np.isfinite(ref_sp[i]).all() and np.isfinite(gen_sp[j]).all()):
            continue

        diff = ref_sp[i] - gen_sp[j]
        sq_err = np.mean(diff ** 2)

        # 計算結果もチェック
        if np.isfinite(sq_err):
            sq_err_sum += sq_err
            valid_count += 1

    if valid_count == 0:
        return 0.0

    return sq_err_sum / valid_count


def calculate_ap_mse_with_alignment(ref_ap, gen_ap, path):
    """アライメントパスを使って非周期性指標(AP)のMSEを計算する

    Args:
        ref_ap: リファレンス音声の非周期性指標
        gen_ap: 生成音声の非周期性指標
        path: DTWアライメントパス

    Returns:
        float: APのMSE
    """
    sq_err_sum = 0.0
    valid_count = 0

    for i, j in path:
        # インデックスチェック
        if i >= len(ref_ap) or j >= len(gen_ap):
            continue

        # NaNやInfをスキップ（PyWorldのd4cは数値的に不安定な場合がある）
        if not (np.isfinite(ref_ap[i]).all() and np.isfinite(gen_ap[j]).all()):
            continue

        diff = ref_ap[i] - gen_ap[j]
        sq_err = np.mean(diff ** 2)

        # 計算結果もチェック
        if np.isfinite(sq_err):
            sq_err_sum += sq_err
            valid_count += 1

    if valid_count == 0:
        return 0.0

    return sq_err_sum / valid_count


def main(args):
    ref_dir = Path(args.ref_dir)    # 正解データ (Whisper)
    gen_dir = Path(args.gen_dir)    # 生成データ (Pseudo)
    output_csv = Path(args.output_csv)

    # 検索対象のファイルリスト (生成データを基準に検索)
    gen_files = sorted(list(gen_dir.rglob("*.wav")))

    print(f"Processing {len(gen_files)} files...")

    # CSVヘッダー書き込み (F0, SP, APのMSEを追加)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'MCD_dB', 'Mel_MSE',
            'F0_MSE', 'LogF0_RMSE', 'SP_MSE', 'AP_MSE'
        ])

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

                # MCD計算
                mcd = calculate_mcd_with_dtw(ref_wav, gen_wav, args.sr)

                # Mel MSE計算 (アライメントパスも取得)
                mel_mse, alignment_path = calculate_mel_mse_with_dtw(
                    ref_wav, gen_wav, args.sr
                )

                # WORLD特徴量抽出（メルスペクトログラムと同じフレーム周期）
                ref_f0, ref_sp, ref_ap = extract_world_features(
                    ref_wav, args.sr, hop_length=config.hop_length
                )
                gen_f0, gen_sp, gen_ap = extract_world_features(
                    gen_wav, args.sr, hop_length=config.hop_length
                )

                # メルスペクトログラムのアライメントを使って
                # F0, SP, APのMSEを計算
                f0_mse = calculate_f0_mse_with_alignment(
                    ref_f0, gen_f0, alignment_path
                )
                log_f0_mse = calculate_log_f0_rmse_with_alignment(
                    ref_f0, gen_f0, alignment_path
                )
                sp_mse = calculate_sp_mse_with_alignment(
                    ref_sp, gen_sp, alignment_path
                )
                ap_mse = calculate_ap_mse_with_alignment(
                    ref_ap, gen_ap, alignment_path
                )

                # CSVに書き込み
                writer.writerow([
                    gen_path.name,
                    f"{mcd:.4f}",
                    f"{mel_mse:.4f}",
                    f"{f0_mse:.4f}",
                    f"{log_f0_mse:.4f}",
                    f"{sp_mse:.4f}",
                    f"{ap_mse:.4f}"
                ])

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
