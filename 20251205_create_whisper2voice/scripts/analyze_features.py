import argparse
import csv
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from scipy.fftpack import dct
from scipy.spatial.distance import cdist

from utils import functions
from utils import config


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
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=None
    )
    mel_sp = np.dot(mel_basis, sp)

    # 対数変換
    log_mel = functions.dynamic_range_compression(mel_sp)
    return log_mel.T  # (Time, n_mels)


def calculate_mcd_with_dtw(ref_wav, gen_wav, sr, n_mcep=13):
    """
    DTWを用いてアライメントを取り、MCD (dB) を計算する。
    acoustic_analysis.pyと同じ方法でDCTを使用して
    メルケプストラム係数を計算。

    Parameters:
        ref_wav: 参照音声波形
        gen_wav: 生成音声波形
        sr: サンプリングレート
        n_mcep: メルケプストラム係数の次数（デフォルト13）

    Returns:
        float: MCD値 [dB]
    """
    # 1. 対数メルスペクトログラムを抽出
    ref_mel = compute_log_mel(ref_wav, sr)  # (Time, n_mels)
    gen_mel = compute_log_mel(gen_wav, sr)  # (Time, n_mels)

    # 2. DTWによるアライメント計算
    # コスト行列を計算（ユークリッド距離）
    C = cdist(ref_mel, gen_mel, metric='euclidean')

    # librosaのDTWを実行してアライメントパスを取得
    D, wp = librosa.sequence.dtw(C=C, backtrack=True)

    # アライメントパスに基づいて両方のスペクトログラムを整列
    ref_mel_aligned = ref_mel[wp[:, 0]]
    gen_mel_aligned = gen_mel[wp[:, 1]]

    # 3. DCT-IIを適用してメルケプストラム係数を計算
    # 対数メルスペクトログラムは既に対数圧縮されているので直接DCTを適用
    ref_mcc = dct(ref_mel_aligned, type=2, axis=1, norm='ortho')[:, :n_mcep]
    gen_mcc = dct(gen_mel_aligned, type=2, axis=1, norm='ortho')[:, :n_mcep]

    # 4. MCDを計算: 10/ln(10) * sqrt(2 * sum((mcc1 - mcc2)^2))
    # 通常は0次成分（パワー）を除外する
    K = 10.0 / np.log(10.0) * np.sqrt(2.0)
    mcd = K * np.mean(
        np.sqrt(np.sum((ref_mcc[:, 1:] - gen_mcc[:, 1:]) ** 2, axis=1))
    )

    return float(mcd)


def calculate_mel_mse_with_dtw(ref_wav, gen_wav, sr):
    """
    DTWを用いてアライメントを取り、メルスペクトログラムのMSEを計算する。
    acoustic_analysis.pyと同じ方法を使用。
    """
    # 1. 特徴量抽出 (Log Mel Spectrogram)
    ref_mel = compute_log_mel(ref_wav, sr)  # (Time, n_mels)
    gen_mel = compute_log_mel(gen_wav, sr)  # (Time, n_mels)

    # 2. DTWによるアライメント
    # コスト行列を計算（ユークリッド距離）
    C = cdist(ref_mel, gen_mel, metric='euclidean')

    # librosaのDTWを実行してアライメントパスを取得
    D, wp = librosa.sequence.dtw(C=C, backtrack=True)

    # アライメントパスに基づいて両方のスペクトログラムを整列
    ref_mel_aligned = ref_mel[wp[:, 0]]
    gen_mel_aligned = gen_mel[wp[:, 1]]

    # 3. MSE (Mean Squared Error) の計算
    mse = np.mean((ref_mel_aligned - gen_mel_aligned) ** 2)

    return float(mse)


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
        default=24000,
        help="Sampling rate"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    main(args)
