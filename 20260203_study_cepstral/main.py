"""
main.py - ケプストラム分析のエントリーポイント

このスクリプトは、音声データからケプストラムを計算する
機能をデモンストレーションします。
"""

from pathlib import Path
from cepstrum_analysis import (
    calculate_cepstrum,
    plot_cepstrum,
    extract_pitch_from_cepstrum,
    process_audio_dataset
)
from visualize_spectrum_cepstrum import (
    visualize_power_spectrum_and_cepstrum
)
from pyworld_mel_cepstral import (
    analyze_single_audio
)


def example_single_file_cepstrum():
    """
    単一ファイルのケプストラム計算の例

    datasetフォルダから1つの音声ファイルを読み込み、
    ケプストラムを計算して可視化します。
    """
    print("=" * 60)
    print("単一ファイルのケプストラム分析")
    print("=" * 60)

    # データセットフォルダ内の音声ファイルを探す
    dataset_dir = Path("dataset/nonpara30/jvs001/wav")

    # 音声ファイルを検索
    audio_files = list(dataset_dir.glob("*.wav"))

    if not audio_files:
        print("エラー: データセットに音声ファイルが見つかりません。")
        return

    # 最初の音声ファイルを使用
    audio_path = audio_files[0]
    print(f"\n使用する音声ファイル: {audio_path}")

    # ケプストラムを計算
    print("\nケプストラムを計算中...")
    cepstrum, quefrency, sr = calculate_cepstrum(audio_path)

    print(f"サンプリングレート: {sr} Hz")
    print(f"ケプストラム形状: {cepstrum.shape}")

    # 結果を可視化
    output_dir = Path("results/cepstrum")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{audio_path.stem}_cepstrum.png"

    print("\nケプストラムを可視化中...")
    plot_cepstrum(cepstrum, quefrency, sr, save_path=save_path)

    # ピッチ推定
    print("\nピッチを推定中...")
    f0_estimates = extract_pitch_from_cepstrum(cepstrum, sr)
    print("推定F0の統計:")
    print(f"  - 平均: {f0_estimates.mean():.2f} Hz")
    print(f"  - 最小: {f0_estimates.min():.2f} Hz")
    print(f"  - 最大: {f0_estimates.max():.2f} Hz")

    print("\n処理完了!")


def example_batch_processing():
    """
    データセット一括処理の例

    datasetフォルダ内のすべての音声ファイルに対して
    ケプストラム分析を実行します。
    """
    print("=" * 60)
    print("データセット一括処理")
    print("=" * 60)

    dataset_dir = Path("dataset")
    output_dir = Path("results/cepstrum_batch")

    if not dataset_dir.exists():
        print(f"エラー: データセットディレクトリが見つかりません: {dataset_dir}")
        return

    print(f"\nデータセットディレクトリ: {dataset_dir}")
    print(f"出力ディレクトリ: {output_dir}")

    # 一括処理を実行
    process_audio_dataset(dataset_dir, output_dir)

    print("\n一括処理完了!")


def example_custom_parameters():
    """
    カスタムパラメータでのケプストラム計算の例

    FFTサイズやホップ長などのパラメータを
    カスタマイズしてケプストラムを計算します。
    """
    print("=" * 60)
    print("カスタムパラメータでのケプストラム分析")
    print("=" * 60)

    # データセットから音声ファイルを取得
    dataset_dir = Path("dataset")
    audio_files = list(dataset_dir.glob("**/*.wav"))

    if not audio_files:
        print("エラー: データセットに音声ファイルが見つかりません。")
        return

    audio_path = audio_files[0]
    print(f"\n使用する音声ファイル: {audio_path}")

    # カスタムパラメータ
    n_fft = 4096  # より高い周波数分解能
    hop_length = 256  # より細かい時間分解能

    print("\nパラメータ:")
    print(f"  - FFTサイズ: {n_fft}")
    print(f"  - ホップ長: {hop_length}")

    # ケプストラムを計算
    cepstrum, quefrency, sr = calculate_cepstrum(
        audio_path,
        n_fft=n_fft,
        hop_length=hop_length
    )

    print(f"\nケプストラム形状: {cepstrum.shape}")

    # 結果を保存
    output_dir = Path("results/cepstrum_custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{audio_path.stem}_custom_cepstrum.png"

    plot_cepstrum(cepstrum, quefrency, sr, save_path=save_path)

    print("\n処理完了!")


def example_visualize_spectrum_cepstrum():
    """
    パワースペクトルと実ケプストラムの可視化の例

    音声ファイルからパワースペクトルと実ケプストラムを計算し、
    それぞれを可視化します。
    """
    print("=" * 60)
    print("パワースペクトルと実ケプストラムの可視化")
    print("=" * 60)

    # データセットフォルダ内の音声ファイルを探す
    dataset_dir = Path("dataset/nonpara30/jvs001/wav")

    # 音声ファイルを検索
    audio_files = list(dataset_dir.glob("*.wav"))

    if not audio_files:
        print("エラー: データセットに音声ファイルが見つかりません。")
        return

    # 最初の音声ファイルを使用
    audio_path = audio_files[0]
    print(f"\n使用する音声ファイル: {audio_path}")

    # 出力ディレクトリを設定
    output_dir = Path("results/spectrum_cepstrum")

    # パワースペクトルと実ケプストラムを可視化
    power_spectrum, real_cepstrum, sr = (
        visualize_power_spectrum_and_cepstrum(
            audio_path,
            frame_length=2048,
            hop_length=512,
            save_dir=output_dir
        )
    )

    print("\n処理完了!")
    print(f"結果は {output_dir} に保存されました。")


def example_pyworld_mel_cepstrum():
    """
    PyWorldを使ったメルケプストラム分析の例

    pyworldを使用してメルケプストラムを計算し、
    librosаで可視化します。
    """
    print("=" * 60)
    print("PyWorld メルケプストラム分析")
    print("=" * 60)

    # データセットフォルダ内の音声ファイルを探す
    dataset_dir = Path("dataset/nonpara30/jvs001/wav")

    # 音声ファイルを検索
    audio_files = list(dataset_dir.glob("*.wav"))

    if not audio_files:
        print("エラー: データセットに音声ファイルが見つかりません。")
        return

    # 最初の音声ファイルを使用
    audio_path = audio_files[0]
    print(f"\n使用する音声ファイル: {audio_path}")

    # 出力ディレクトリを設定
    output_dir = Path("results/mel_cepstrum")

    # メルケプストラム分析を実行
    analyze_single_audio(audio_path, output_dir)

    print("\n処理完了!")
    print(f"結果は {output_dir} に保存されました。")


def main():
    """
    メインエントリーポイント

    各種ケプストラム分析の例を実行します。
    """
    print("\n" + "=" * 60)
    print("ケプストラム分析デモンストレーション")
    print("=" * 60 + "\n")

    # 例1: PyWorldメルケプストラム分析
    example_pyworld_mel_cepstrum()

    print("\n")

    # 例2: 単一ファイルのケプストラム計算（コメントアウト）
    # example_single_file_cepstrum()

    # 例3: カスタムパラメータでの計算（コメントアウト）
    # example_custom_parameters()

    # 例4: データセット一括処理（コメントアウト）
    # 注意: データセット内のファイル数が多い場合は時間がかかります
    # example_batch_processing()


if __name__ == "__main__":
    main()
