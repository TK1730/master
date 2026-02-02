import argparse
import numpy as np
import librosa
import pyworld as pw
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def create_topline_dataset(input_dir, output_dir, sr=24000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # wavファイルを取得
    wav_files = sorted(list(input_dir.rglob("*.wav")))
    print(f"Processing {len(wav_files)} files from {input_dir}...")

    for wav_path in tqdm(wav_files):
        # 読み込み
        x, _ = librosa.load(wav_path, sr=sr)
        x = x.astype(np.float64)

        try:
            # PyWorldで分析 (Analysis)
            # whisperのような無声音や息漏れが多い音は harvest が安定します
            f0, t = pw.harvest(x, sr)
            sp = pw.cheaptrick(x, f0, t, sr)
            ap = pw.d4c(x, f0, t, sr)

            # そのまま再合成 (Resynthesis)
            y = pw.synthesize(f0, sp, ap, sr)

            # 保存
            # 元のディレクトリ構造を維持して保存
            rel_path = wav_path.relative_to(input_dir)
            save_path = output_dir / rel_path

            save_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(save_path, y, sr, subtype="PCM_16")

        except Exception as e:
            print(f"Error processing {wav_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 既存のwhisper10のパス (適宜書き換えてください)
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to original whisper10 dataset")
    # 出力先
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output path for resynthesized audio")
    parser.add_argument("--sr", type=int, default=22050, help="Sampling rate")

    args = parser.parse_args()

    create_topline_dataset(args.input_dir, args.output_dir, args.sr)
