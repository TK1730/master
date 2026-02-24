"""
メインエントリーポイント

このスクリプトは各種機能のデモンストレーションを提供します。
"""

from pathlib import Path
from scripts.dtw_alignment import (
    compare_audio_with_dtw,
    load_audio,
    compute_mel_spectrogram,
    align_spectrograms_with_dtw,
    compute_mse_difference,
    visualize_difference_only
)


def example_dtw_alignment():
    """
    DTWアライメントと差分可視化のサンプル

    2つの音声ファイルのメルスペクトログラムを計算し、
    DTWでアライメントを取り、差分を可視化します。
    """
    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent

    # サンプル音声ファイルのパス
    # 注: 実際のファイルパスに置き換えてください
    audio1_path = (
        project_root /
        "dataset/nonpara30w_ver2/jvs001/wav/BASIC5000_0235.wav"
    )
    audio2_path = (
        project_root /
        "dataset/whisper10/jvs001/wav/BASIC5000_0235.wav"
    )

    # 出力パス
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "dtw_alignment_comparison.png"

    # 音声ファイルの存在確認
    if not audio1_path.exists():
        print(f"警告: {audio1_path} が見つかりません")
        print("サンプルのために別のファイルパスを指定してください")
        return

    if not audio2_path.exists():
        print(f"警告: {audio2_path} が見つかりません")
        print("サンプルのために別のファイルパスを指定してください")
        return

    print("=" * 60)
    print("DTW音声アライメントと差分可視化のデモ")
    print("=" * 60)

    # DTWアライメントと差分可視化を実行
    compare_audio_with_dtw(
        audio_path1=str(audio1_path),
        audio_path2=str(audio2_path),
        output_path=str(output_path),
        sr=22050,
        n_mels=128,
        hop_length=512
    )


def example_difference_spectrogram_only():
    """
    差分スペクトログラムのみを保存するサンプル

    2つの音声ファイルのメルスペクトログラムを計算し、
    DTWでアライメントを取り、MSE差分スペクトログラムのみを保存します。
    """
    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent

    # サンプル音声ファイルのパス
    audio1_path = (
        project_root /
        "dataset/nonpara30w_ver2/jvs001/wav/BASIC5000_0235.wav"
    )
    audio2_path = (
        project_root /
        "dataset/whisper10/jvs001/wav/BASIC5000_0235.wav"
    )

    # 出力パス
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "mse_difference_spectrogram.png"

    # 音声ファイルの存在確認
    if not audio1_path.exists():
        print(f"警告: {audio1_path} が見つかりません")
        return

    if not audio2_path.exists():
        print(f"警告: {audio2_path} が見つかりません")
        return

    print("=" * 60)
    print("MSE差分スペクトログラム可視化のデモ")
    print("=" * 60)

    # 音声ファイルを読み込み
    print(f"音声ファイル1を読み込み中: {audio1_path.name}")
    y1, sr1 = load_audio(str(audio1_path), sr=22050)

    print(f"音声ファイル2を読み込み中: {audio2_path.name}")
    y2, sr2 = load_audio(str(audio2_path), sr=22050)

    # メルスペクトログラムを計算
    print("メルスペクトログラムを計算中...")
    spec1 = compute_mel_spectrogram(y1, sr1, n_mels=80, hop_length=256)
    spec2 = compute_mel_spectrogram(y2, sr2, n_mels=80, hop_length=256)

    # DTWアライメントを実行
    print("DTWアライメントを実行中...")
    dtw_matrix, alignment_path = align_spectrograms_with_dtw(spec1, spec2)

    # MSE差分を計算
    print("MSE差分を計算中...")
    diff_spec, mse = compute_mse_difference(spec1, spec2, alignment_path)
    print(f"計算されたMSE: {mse:.4f} dB²")

    # 差分スペクトログラムのみを可視化
    print("差分スペクトログラムを可視化中...")
    visualize_difference_only(
        diff_spec=diff_spec,
        sr=sr1,
        hop_length=256,
        audio1_name=audio1_path.stem,
        audio2_name=audio2_path.stem,
        mse=mse,
        output_path=str(output_path),
        figsize=(14, 7)
    )

    print("処理完了!")


def main():
    """
    メインエントリーポイント
    """
    # 差分スペクトログラムのみを保存するサンプルを実行
    example_difference_spectrogram_only()

    # 全体の比較画像が必要な場合は以下をコメント解除
    # example_dtw_alignment()


if __name__ == "__main__":
    main()
