import os
import librosa
import soundfile as sf


def normalize_and_save(
    input_path,
    output_filename,
    sample_rate=None,
    output_dir="material"
):
    """
    音声ファイルを読み込み、-1から1に正規化してWAVファイルとして保存する。

    Parameters
    ----------
    input_path : str
        入力音声ファイルのパス
    output_filename : str
        出力ファイル名（拡張子を含む）
    sample_rate : int, optional
        サンプリングレート。Noneの場合は元のサンプリングレートを使用
    output_dir : str, optional
        出力ディレクトリのパス（デフォルト: "material"）

    Returns
    -------
    output_path : str
        保存したファイルの絶対パス
    """
    # 音声ファイルを読み込み
    y, sr = librosa.load(input_path, sr=sample_rate)

    # 正規化前の範囲を確認
    print(f"正規化前の範囲: [{y.min():.6f}, {y.max():.6f}]")

    # -1から1に正規化（ピーク値を±1.0にする）
    y_normalized = librosa.util.normalize(y)

    # 正規化後の範囲を確認
    print(f"正規化後の範囲: [{y_normalized.min():.6f}, "
          f"{y_normalized.max():.6f}]")

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 出力パスを作成
    output_path = os.path.join(output_dir, output_filename)

    # 正規化した音声をWAVファイルとして保存
    sf.write(output_path, y_normalized, sr)

    print(f"正規化した音声を保存しました: {os.path.abspath(output_path)}")

    return os.path.abspath(output_path)


if __name__ == "__main__":
    # 使用例: 音声を正規化してmaterialディレクトリに保存
    input_file = [
        "dataset/nonpara30/jvs057/wav/UT-PARAPHRASE-sent045-phrase1.wav",
        "dataset/nonpara30w_ver2/jvs057/wav/UT-PARAPHRASE-sent045-phrase1.wav",
        "dataset/whisper_converted_v2/jvs057/wav/UT-PARAPHRASE-sent045-phrase1.wav",
        "dataset/whisper2voice/whisper_converted_v2/jvs057/wav/UT-PARAPHRASE-sent045-phrase1.wav",
        "dataset/whisper2voice/whisper2voice/jvs057/wav/UT-PARAPHRASE-sent045-phrase1.wav"
    ]
    output_file = [
        "normalized_nonpara30_UT-PARAPHRASE-sent045-phrase1.wav",
        "normalized_nonpara30w_ver2_UT-PARAPHRASE-sent045-phrase1.wav",
        "normalized_whisper_converted_v2_UT-PARAPHRASE-sent045-phrase1.wav",
        "normalized_whisper2voice_whisper_converted_v2_UT-PARAPHRASE-sent045-phrase1.wav",
        "normalized_whisper2voice_whisper2voice_UT-PARAPHRASE-sent045-phrase1.wav"
    ]

    for input, output in zip(input_file, output_file):
        normalize_and_save(
            input_path=input,
            output_filename=output,
            sample_rate=22050
        )
