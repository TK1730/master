from pathlib import Path
from typing import Optional


def get_transcription_jvs_path(
    spk_transcription_path: Path,
    directory: str,
) -> Optional[Path]:
    """jvsデータセットのトランスクリプションファイルのパスを取得する

    Args:
        spk_transcription_path (Path): transcriptionファイルの話者ディレクトリのパス
        directory (str): 話者ディレクトリ内のサブディレクトリ名

    Returns:
        Path: トランスクリプションファイルのパス
    """
    transcription_path = spk_transcription_path / directory
    # glob の最初の .txt を返す。なければ None を返す
    txt_iter = transcription_path.glob("*.txt")
    transcription = next(txt_iter, None)
    if transcription is None:
        print(f"No transcription file found: {transcription_path}")
        return None
    return transcription


def rewrite_transcription_jvs(
    transcription_path: Path,
    output_file: Path,
    speaker_id: str,
    language: str = "JP",
) -> None:
    """jvsデータセットのトランスクリプションファイルを読み込み、vits2用に書き換える

    Args:
        transcription_path (Path): トランスクリプションファイルのパス (例: .../jvs001/nonpara30/transcript.txt)
        output_path (Path): 書き換えたトランスクリプションファイルの出力ベースパス (例: preprocessed/jvs_ver1)
        speaker_id (str): 話者ID
        language (str, optional): 言語コード. Defaults to "JP".
    """
    output_lines = []

    # If no transcription path is provided, skip
    if transcription_path is None:
        print(f"No transcription file provided: {transcription_path}")
        return

    # transcription_path はファイルパスなので parent が nonpara30, parent.parent が jvs001
    try:
        subdir = transcription_path.parent.name
        spk_dir = transcription_path.parent.parent.name
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
            wav_name = f"{file}.wav"

            # output 側で期待する wav のパス
            expected_wav = (
                output_file.parent / spk_dir / subdir / "wav24kHz16bit" / wav_name
            )

            if expected_wav.exists():
                # Ensure speaker_id is a string
                processed_line = (
                    f"{expected_wav.as_posix()}|{str(speaker_id)}|{language}|{text}\n"
                )
                output_lines.append(processed_line)
                print(f"rel_path: {expected_wav.as_posix()}")

    # Write/appending processed lines into a single transcription file under output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_lines:
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(output_lines)
    print(f"Wrote {len(output_lines)} entries to {output_file}")


if __name__ == "__main__":
    # TODO: どのファイルでも使えるようにする
    # 理想：dataset_path, output_pathを引数で受け取るだけで動くようにする
    dataset_path = Path("jvs_ver1/jvs_ver1")
    output_path = Path("preprocessed/jvs_ver1/transcription_whisper10.txt")
    # jvsデータセットの各話者ディレクトリを走査
    for speaker_dir in dataset_path.iterdir():
        if speaker_dir.is_dir():
            jvs_speaker_id = speaker_dir.name
            speaker_id = f"{jvs_speaker_id}_whisper10"
            language = "JP"
            transcription = get_transcription_jvs_path(speaker_dir, "whisper10")
            rewrite_transcription_jvs(transcription, output_path, speaker_id, language)
