# -*- coding: utf-8 -*-
"""
メルスペクトログラム可視化スクリプト

入力データ(Ref)、ターゲット(Target)、生成データ(Gen)のメルスペクトログラムを
1枚の画像に並べて描画・保存します。
"""
import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートのパスを追加 (noqa: E402)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402


def compute_mel_spectrogram(
    audio_path: str,
    sr: int = config.sr,
    n_fft: int = config.n_fft,
    hop_length: int = config.hop_length,
    n_mels: int = config.n_mels,
    fmin: float = config.fmin,
    fmax: float = config.fmax
) -> tuple[np.ndarray, int]:
    """
    音声ファイルからメルスペクトログラムを計算する。

    Args:
        audio_path: 音声ファイルのパス
        sr: サンプリングレート
        n_fft: FFTサイズ
        hop_length: ホップ長
        n_mels: メルバンド数
        fmin: 最低周波数
        fmax: 最高周波数

    Returns:
        tuple: (対数メルスペクトログラム, サンプリングレート)
    """
    # 音声読み込み
    wav, sr_loaded = librosa.load(audio_path, sr=sr)

    # メルスペクトログラム計算
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    # 対数変換 (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec, sr_loaded


def visualize_mel_spectrograms(
    ref_path: str,
    target_path: str,
    gen_path: str,
    output_path: str,
    sr: int = config.sr,
    hop_length: int = config.hop_length,
    max_time: float = None
) -> None:
    """
    3つの音声ファイルのメルスペクトログラムを個別の画像として保存する。

    Args:
        ref_path: 入力データ(参照)の音声ファイルパス
        target_path: ターゲットの音声ファイルパス
        gen_path: 生成データの音声ファイルパス
        output_path: 出力画像パス（ベース名として使用）
        sr: サンプリングレート
        hop_length: ホップ長
        max_time: 表示する最大時間（秒）。Noneの場合は全体を表示
    """
    # 日本語フォント設定 (Windows環境)
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # 各音声のメルスペクトログラムを計算
    ref_mel, _ = compute_mel_spectrogram(ref_path)
    target_mel, _ = compute_mel_spectrogram(target_path)
    gen_mel, _ = compute_mel_spectrogram(gen_path)

    # 時間制限がある場合、フレーム数を計算
    if max_time is not None:
        max_frames = int(max_time * sr / hop_length)
        ref_mel = ref_mel[:, :max_frames]
        target_mel = target_mel[:, :max_frames]
        gen_mel = gen_mel[:, :max_frames]

    # 共通のカラーマップ範囲
    vmin = min(ref_mel.min(), target_mel.min(), gen_mel.min())
    vmax = max(ref_mel.max(), target_mel.max(), gen_mel.max())

    # 出力パスのベース名を取得
    output_base = Path(output_path)
    output_dir = output_base.parent
    stem = output_base.stem

    # データと設定のリスト
    mel_data = [
        (ref_mel, 'Input (Ref): Whisper Voice', f'{stem}_ref.png'),
        (target_mel, 'Target: Pseudo-Whisper Voice', f'{stem}_target.png'),
        (gen_mel, 'Generated (Gen): VITS Converted', f'{stem}_gen.png'),
    ]

    # 各メルスペクトログラムを個別に保存
    for mel_spec, title, filename in mel_data:
        fig, ax = plt.subplots(figsize=(10, 4))

        img = librosa.display.specshow(
            mel_spec,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap='magma'
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

        plt.tight_layout()
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"保存: {save_path}")


def find_matching_files(
    gen_dir: Path,
    target_dir: Path,
    ref_dir: Path
) -> tuple[Path, Path, Path] | None:
    """
    最初にマッチするファイルセットを探す。

    Args:
        gen_dir: 生成音声のディレクトリ
        target_dir: ターゲット音声のディレクトリ
        ref_dir: 参照音声のディレクトリ

    Returns:
        tuple: (ref_path, target_path, gen_path) または None
    """
    gen_files = sorted(list(gen_dir.rglob("*.wav")))

    for gen_path in gen_files:
        rel_path = gen_path.relative_to(gen_dir)

        # 対応するファイルを探す
        target_path = target_dir / rel_path
        ref_path = ref_dir / rel_path

        # 同じ相対パスで見つからない場合はファイル名で検索
        if not target_path.exists():
            candidates = list(target_dir.rglob(gen_path.name))
            if candidates:
                target_path = candidates[0]
            else:
                continue

        if not ref_path.exists():
            candidates = list(ref_dir.rglob(gen_path.name))
            if candidates:
                ref_path = candidates[0]
            else:
                continue

        return ref_path, target_path, gen_path

    return None


def main(args: argparse.Namespace) -> None:
    """
    メイン処理。

    Args:
        args: コマンドライン引数
    """
    target_dir = Path(args.target_dir)
    gen_dir = Path(args.gen_dir)
    ref_dir = Path(args.ref_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # マッチするファイルを探す
    matched = find_matching_files(gen_dir, target_dir, ref_dir)

    if matched is None:
        print("マッチするファイルが見つかりませんでした。")
        return

    ref_path, target_path, gen_path = matched

    print("使用ファイル:")
    print(f"  Ref (Input):    {ref_path}")
    print(f"  Target:         {target_path}")
    print(f"  Gen (Output):   {gen_path}")

    # 出力ファイル名
    output_file = output_dir / f"mel_spectrogram_{gen_path.stem}.png"

    # 可視化 (2.4秒までの範囲で個別保存)
    visualize_mel_spectrograms(
        ref_path=str(ref_path),
        target_path=str(target_path),
        gen_path=str(gen_path),
        output_path=str(output_file),
        max_time=2.4
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="メルスペクトログラムを可視化して保存します"
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="生成音声 (VITS) のディレクトリ"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="ターゲット音声 (疑似ささやき声) のディレクトリ"
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        required=True,
        help="参照音声 (元のささやき声) のディレクトリ"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="出力ディレクトリ"
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
