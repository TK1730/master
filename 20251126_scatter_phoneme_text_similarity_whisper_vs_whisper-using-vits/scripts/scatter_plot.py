import argparse
import csv
from pathlib import Path
from typing import Union, Dict

import numpy as np
import torch
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.lstm_net_rev import LSTM_net
from utils import config, functions


def scatter_plot(
    x: list,
    y: list,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Union[str, Path]
):
    """散布図を作成して保存する関数"""
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.6)

    # 傾向線（Trend line）を引くと分析しやすい
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trand = np.linspace(0, 1, 100)
        plt.plot(x_trand, p(x_trand), "r--", alpha=0.8, label="Trend")
        plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    # スケーリングを合わせる (0.0 - 1.0)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    # アスペクト比を1:1に固定 (正方形プロット)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_phoneme_model_inf(
    model_path: Union[str, Path],
    device: str = "cpu"
) -> LSTM_net:
    model = LSTM_net(
        n_inputs=80,
        n_outputs=36,
        n_layers=2,
        hidden_size=128,
        fc_size=4096,
        dropout=0.2,
        bidirectional=True
    )
    model_path, _ = functions.best_model(model_path)
    model_info = torch.load(model_path, map_location=device)
    model.load_state_dict(model_info)
    model.eval()
    return model


def recognize_phoneme_indices(
    model: LSTM_net,
    msp: np.ndarray,
    device: str
) -> tuple[np.ndarray, np.ndarray]:
    # msp: (n_mels, Time) -> model input: (Batch, Time, n_mels)
    msp_tensor = torch.tensor(msp, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(msp_tensor)
        probs = torch.softmax(output, dim=-1)
        # 最大値のインデックスを取得
        max_idx = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()
    return max_idx


def calculate_match_rate(
    seq1: np.ndarray,
    seq2: np.ndarray
) -> float:
    """2つの音素列の一致率を計算する関数"""
    if len(seq1) != len(seq2):
        raise ValueError(
            "Sequences must be of the same length to calculate match rate.")
    matches = np.sum(seq1 == seq2)
    return matches / len(seq1)


def load_similarity_csv(
    csv_path: str,
    key_col: str = 'file',
    val_col: str = 'ratio'
) -> Dict[str, float]:
    """
    文章類似度が記載されたCSVを読み込み、辞書 {ファイル名: 類似度} を返す。
    想定データ: ./data/converted_whisper2voice_v2_transcription_text.csv
    """
    df = pd.read_csv(csv_path)

    # カラム名チェック (存在しない場合は列番号で仮定するかエラーにする)
    if key_col not in df.columns:
        # もし 'file' カラムがない場合、パスが含まれるカラムを探すなどの処理が必要だが
        # ここではユーザー指定のカラムを必須とする
        print(f"Warning: Column '{key_col}' not found in CSV."
              f" Available columns: {df.columns}")
        return {}

    if val_col not in df.columns:
        # 'ratio'がない場合、それっぽいカラムを探す（cosine_similarityなど）
        cols = [c for c in df.columns if 'sim' in c.lower()]
        if cols:
            val_col = cols[0]
            print(f"Using '{val_col}' as similarity column.")
        else:
            print(f"Warning: Column '{val_col}' not found in CSV.")
            return {}

    # 辞書作成 (ファイル名 -> 類似度)
    # ファイル名がパス付きの場合は basename をキーにするなど工夫が必要
    # ここではデータソースに合わせて柔軟に対応するため、basenameをキーにする
    sim_dict = {}
    for _, row in df.iterrows():
        fname = str(row[key_col])
        # パス
        fname_key = Path(fname)
        try:
            val = float(row[val_col])
            sim_dict[fname_key] = val
        except ValueError:
            continue

    return sim_dict


def wav2msp_inference(wav_path: Union[str, Path], sr=22050) -> np.ndarray:
    """wavファイルからメルスペクトログラムに変換する関数"""
    y, _ = librosa.load(wav_path, sr=sr)
    D = librosa.stft(
        y=y,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        pad_mode='reflect',
        center=False
    )
    sp, phase = librosa.magphase(D)  # 振幅スペクトル 位相スペクトル
    msp = np.matmul(sp.T, config.mel_filter.T)  # メルスペクトルを抽出
    lmsp = functions.dynamic_range_compression(msp)

    return lmsp


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Step 1: モデルロード
    print("Step 1: Loading phoneme recognition model...")
    model = load_phoneme_model_inf(args.model_path, device)

    # 2. Step 3: 文章類似度データのロード
    print("Step 3: Loading text similarity data...")
    sim_dict = load_similarity_csv(
        args.sim_csv,
        args.csv_key_col,
        args.csv_val_col
    )
    print(f"Loaded {len(sim_dict)} similarity records.")

    ref_dir = Path(args.ref_dir)
    gen_dir = Path(args.gen_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Step 2 & 4: 音素一致率算出と散布図作成
    gen_files = sorted(list(gen_dir.rglob("*.wav")))
    x_phoneme_rates = []
    y_text_sims = []
    filenames = []

    phoneme_match_results = {}

    print(f"Processing {len(gen_files)} files...")
    for gen_path in tqdm(gen_files):
        filename = Path(gen_path.parts[-3] + "/" + gen_path.stem)
        # print(filename)

        if filename not in sim_dict:
            continue

        rel_path = gen_path.relative_to(gen_dir)
        ref_path = ref_dir / rel_path

        if not ref_path.exists():
            candidates = list(ref_dir.rglob(filename))
            if candidates:
                ref_path = candidates[0]
            else:
                continue

        try:
            # --- 音素認識 (推論) ---
            msp_gen = wav2msp_inference(gen_path, sr=args.sr)
            msp_ref = wav2msp_inference(ref_path, sr=args.sr)

            indices_gen = recognize_phoneme_indices(model, msp_gen, device)
            indices_ref = recognize_phoneme_indices(model, msp_ref, device)

            # --- 一致率計算 ---
            # DTWで長さが揃っているため、そのまま比較可能
            match_rate = calculate_match_rate(
                indices_ref, indices_gen)

            phoneme_match_results[str(rel_path)] = match_rate

            text_sim = sim_dict[filename]
            x_phoneme_rates.append(match_rate)
            y_text_sims.append(text_sim)
            filenames.append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 4. Step 2: metrics.csvの保存
    metrics_csv = output_dir / "metrics_phoneme_match.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "phoneme_match_rate"])
        for k, v in phoneme_match_results.items():
            writer.writerow([k, v])
    print(f"Saved phoneme match metrics to {metrics_csv}")

    # 5. 散布図と相関係数の保存
    if len(x_phoneme_rates) > 0:
        print("Step 4: Generating scatter plot and calculating correlation...")
        output_png = output_dir / "scatter_phoneme_vs_text_sim.png"

        # 相関係数の算出 (Pearson)
        correlation = np.corrcoef(x_phoneme_rates, y_text_sims)[0, 1]

        scatter_plot(
            x=x_phoneme_rates,
            y=y_text_sims,
            x_label="Phoneme Match Rate",
            y_label="Text Similarity (ReazonSpeech)",
            title=(
                f"Phoneme Similarity vs Text Similarity\n"
                f"(n={len(x_phoneme_rates)}, r={correlation:.4f})"
            ),
            output_path=output_png
        )
        print(f"Saved scatter plot to {output_png}")

        # 相関係数をCSV保存
        corr_csv = output_dir / "correlation.csv"
        with open(corr_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["pearson_correlation", correlation])
            writer.writerow(["sample_count", len(x_phoneme_rates)])
        print(f"Saved correlation coefficient to {corr_csv}")

        # 散布図の元データもCSV保存
        scatter_csv = output_dir / "scatter_data.csv"
        with open(scatter_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["file", "phoneme_match_rate", "text_similarity"])
            for n, p, t in zip(filenames, x_phoneme_rates, y_text_sims):
                writer.writerow([n, p, t])
    else:
        print("No valid data points found. Check file and CSV content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="Path to Reference Audio Dir (Whisper)"
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Path to Generated Audio Dir (Pseudo/VITS)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained LSTM phoneme model (.pth)"
    )
    parser.add_argument(
        "--sim_csv",
        type=str,
        required=True,
        help="Path to Text Similarity CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory"
    )

    # Settings
    parser.add_argument("--sr", type=int, default=22050, help="Sampling rate")
    parser.add_argument(
        "--csv_key_col",
        type=str,
        default="file",
        help="Column name for file in CSV"
    )
    parser.add_argument(
        "--csv_val_col",
        type=str,
        default="ratio",
        help="Column name for similarity in CSV"
    )

    args = parser.parse_args()
    main(args)
