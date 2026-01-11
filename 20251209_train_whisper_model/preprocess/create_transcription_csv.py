"""
複数のデータセットディレクトリからtranscription.csvを作成するスクリプト

このスクリプトは異なるデータセットディレクトリから転写ファイルを読み取り、
audio_pathとtranscriptionの列を持つ単一のCSVファイルに統合します。
"""

import csv
import pathlib
from typing import List, Tuple


def parse_line(line: str) -> Tuple[str, str]:
    """
    転写ファイルから1行を解析する

    想定されるフォーマット: ファイル名:テキスト

    Args:
        line: 転写ファイルからの1行

    Returns:
        (ファイル名, 転写テキスト) のタプル
    """
    parts = line.strip().split(':', 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""


def process_standard_dataset(
    dataset_dir: pathlib.Path,
    dataset_name: str
) -> List[Tuple[str, str]]:
    """
    標準的なデータセット構造を処理する

    標準的な構造はサブディレクトリ（例：jvs001, jvs002）を持ち、
    それぞれにtranscripts_utf8.txtとwavフォルダが含まれています。

    Args:
        dataset_dir: データセットディレクトリへのパス
        dataset_name: ログ出力用のデータセット名

    Returns:
        (audio_path, transcription) のタプルのリスト
    """
    rows = []
    print(f"\n{dataset_name}を処理中...")

    # サブディレクトリを反復処理
    for speaker_dir in sorted(dataset_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        transcript_file = speaker_dir / 'transcripts_utf8.txt'
        wav_dir = speaker_dir / 'wav'

        if not transcript_file.exists():
            continue

        print(f"  発見: {speaker_dir.name}/transcripts_utf8.txt")

        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    filename, text = parse_line(line)
                    if not filename:
                        continue

                    # .wav拡張子がない場合は追加
                    if not filename.lower().endswith('.wav'):
                        audio_filename = f"{filename}.wav"
                    else:
                        audio_filename = filename

                    audio_path = wav_dir / audio_filename

                    # ファイルが存在するか確認
                    if not audio_path.exists():
                        print(f"  警告: ファイルが見つかりません: {audio_path}")
                        continue

                    # プロジェクトルートからの相対パスを取得
                    rel_path = audio_path.relative_to(pathlib.Path.cwd())
                    rows.append((str(rel_path), text))

        except Exception as e:
            print(f"  エラー処理 {transcript_file}: {e}")

    print(f"  {dataset_name}から{len(rows)}件のエントリを追加")
    return rows


def process_throat_microphone_dataset(
    dataset_dir: pathlib.Path
) -> List[Tuple[str, str]]:
    """
    throat_microphoneデータセット構造を処理する

    このデータセットは異なる構造を持ち、transcripts_utf8.txtが
    ルートディレクトリにあり、音声ファイルは番号付きの
    サブディレクトリ（041, 042など）のwavフォルダ内にあります。

    Args:
        dataset_dir: throat_microphoneディレクトリへのパス

    Returns:
        (audio_path, transcription) のタプルのリスト
    """
    rows = []
    print("\nthroat_microphoneを処理中...")

    transcript_file = dataset_dir / 'transcripts_utf8.txt'

    if not transcript_file.exists():
        print("  警告: transcripts_utf8.txtが見つかりません")
        return rows

    print("  発見: transcripts_utf8.txt")

    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                filename, text = parse_line(line)
                if not filename:
                    continue

                # ファイル名から番号を抽出
                # 想定フォーマット: sXXXdy (XXXが番号部分)
                # 例: s001dy, s026dy
                try:
                    # 's'の後の3桁の数字を抽出
                    file_number = int(filename[1:4])
                    # ファイル番号からディレクトリ番号を計算
                    # 1-25 -> 041, 26-50 -> 042, ...
                    directory_number = 40 + ((file_number - 1) // 25) + 1
                    speaker_id = f"{directory_number:03d}"
                except (ValueError, IndexError):
                    print(f"  警告: ファイル名の形式が不正です: {filename}")
                    continue

                # .wav拡張子がない場合は追加
                if not filename.lower().endswith('.wav'):
                    audio_filename = f"{filename}.wav"
                else:
                    audio_filename = filename

                # パスを構築: throat_microphone/SPEAKER_ID/wav/FILENAME
                audio_path = dataset_dir / speaker_id / 'wav' / audio_filename

                # ファイルが存在するか確認
                if not audio_path.exists():
                    print(f"  警告: ファイルが見つかりません: {audio_path}")
                    continue

                # プロジェクトルートからの相対パスを取得
                rel_path = audio_path.relative_to(pathlib.Path.cwd())
                rows.append((str(rel_path), text))

    except Exception as e:
        print(f"  エラー処理 {transcript_file}: {e}")

    print(f"  throat_microphoneから{len(rows)}件のエントリを追加")
    return rows


def main():
    """すべてのデータセットを処理してCSVを作成するメイン関数"""
    # ディレクトリを定義
    base_dir = pathlib.Path.cwd()
    dataset_dir = base_dir / 'dataset'
    output_csv = dataset_dir / 'transcription.csv'

    print(f"データセットディレクトリ: {dataset_dir}")
    print(f"出力ファイル: {output_csv}")

    if not dataset_dir.exists():
        print(f"\nエラー: データセットディレクトリが見つかりません {dataset_dir}")
        return

    all_rows = []

    # 標準的なデータセットを処理
    standard_datasets = ['nonpara30', 'nonpara30w_ver2', 'whisper10']
    for dataset_name in standard_datasets:
        dataset_path = dataset_dir / dataset_name
        if dataset_path.exists():
            rows = process_standard_dataset(dataset_path, dataset_name)
            all_rows.extend(rows)
        else:
            print(f"\n警告: {dataset_name}が見つかりません、スキップします...")

    # throat_microphoneデータセットを処理（異なる構造）
    throat_mic_path = dataset_dir / 'throat_microphone'
    if throat_mic_path.exists():
        rows = process_throat_microphone_dataset(throat_mic_path)
        all_rows.extend(rows)
    else:
        print("\n警告: throat_microphoneが見つかりません、スキップします...")

    # CSVに書き込み
    print(f"\n{'='*60}")
    print(f"合計エントリ数: {len(all_rows)}")

    if all_rows:
        print(f"{output_csv}に書き込み中...")
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_path', 'transcription'])
            writer.writerows(all_rows)
        print("transcription.csvの作成に成功しました！")
    else:
        print("エントリが見つかりませんでした。CSVファイルは作成されませんでした。")


if __name__ == "__main__":
    main()
