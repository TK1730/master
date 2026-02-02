from pathlib import Path
from typing import Optional
import unicodedata


def normalize_punctuation(text: str) -> str:
    """全角句読点を半角に正規化する

    Args:
        text (str): 正規化対象のテキスト

    Returns:
        str: 正規化後のテキスト
    """
    # Unicode正規化（全角英数字を半角に）
    text = unicodedata.normalize("NFKC", text)
    # 全角句読点を半角に変換
    replacements = {
        "！": "!",
        "？": "?",
        "，": ",",
        "。": ".",
        "、": ",",
        "：": ":",
        "；": ";",
    }

    for full, half in replacements.items():
        text = text.replace(full, half)

    return text


def get_transcription_jvs_path(
    spk_transcription_path: Path,
    filename: str = "transcripts_utf8.txt",
) -> Optional[Path]:
    """VITS_pretrainデータセットのトランスクリプションファイルのパスを取得する

    Args:
        spk_transcription_path (Path): 話者ディレクトリのパス
        filename (str): トランスクリプションファイル名 (デフォルト: "transcripts_utf8.txt")

    Returns:
        Path: トランスクリプションファイルのパス。存在しない場合はNone
    """
    transcription_path = spk_transcription_path / filename
    if not transcription_path.exists():
        print(f"No transcription file found: {transcription_path}")
        return None
    return transcription_path


def rewrite_transcription_jvs(
    transcription_path: Path,
    output_file: Path,
    speaker_id: str,
    language: str = "JP",
) -> None:
    """VITS_pretrainデータセットのトランスクリプションファイルを
    読み込み、vits2用に書き換える

    Args:
        transcription_path (Path): トランスクリプションファイルのパス
            (例: dataset/VITS_pretrain/common_voice_1/transcripts_utf8.txt)
        output_file (Path): 出力ファイルパス
            (例: preprocessed/VITS_pretrain/transcription.txt)
        speaker_id (str): 話者ID
        language (str, optional): 言語コード. Defaults to "JP".
    """
    output_lines = []

    # If no transcription path is provided, skip
    if transcription_path is None:
        print(f"No transcription file provided: {transcription_path}")
        return

    # transcription_path の親ディレクトリが話者ディレクトリ (例: common_voice_1)
    try:
        spk_dir = transcription_path.parent.name
    except Exception:
        print(f"Invalid transcription_path: {transcription_path}")
        return

    with open(transcription_path, "r", encoding="utf-8") as trans_file:
        for line in trans_file:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                print(f"Skipping malformed line: {line}")
                continue
            file, text = line.split(":", 1)
            # テキストを正規化（全角句読点を半角に）
            text = normalize_punctuation(text)
            wav_name = f"{file}.wav"

            # wavファイルは話者ディレクトリの wav フォルダ内にある
            expected_wav = output_file.parent / spk_dir / "wav" / wav_name

            if expected_wav.exists():
                # Ensure speaker_id is a string
                processed_line = (
                    f"{expected_wav.as_posix()}|{str(speaker_id)}|{language}|{text}\n"
                )
                output_lines.append(processed_line)
                print(f"rel_path: {expected_wav.as_posix()}")
            else:
                print(f"Warning: WAV file not found: {expected_wav}")

    # Write/appending processed lines into a single transcription file under output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_lines:
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(output_lines)
    print(f"Wrote {len(output_lines)} entries to {output_file}")


if __name__ == "__main__":
    # TODO: どのファイルでも使えるようにする
    # 理想：dataset_path, output_pathを引数で受け取るだけで動くようにする
    dataset_path = Path("dataset/VITS_pretrain")
    output_path = Path("preprocessed/VITS_pretrain/transcription.txt")
    # jvsデータセットの各話者ディレクトリを走査
    for speaker_dir in dataset_path.iterdir():
        if speaker_dir.is_dir():
            jvs_speaker_id = speaker_dir.name
            speaker_id = jvs_speaker_id
            language = "JP"
            transcription = get_transcription_jvs_path(
                speaker_dir, "transcripts_utf8.txt"
            )
            rewrite_transcription_jvs(transcription, output_path, speaker_id, language)
