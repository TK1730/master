"""
Common Voice データセット処理スクリプト

Common Voice 日本語データセット (mcv-scripted-ja-v24.0) を処理し、
VITS 学習用に話者ごとに音声ファイルを整理するスクリプト。

主な機能:
- validated.tsv を読み込み、話者ごとにグループ化
- 10発話未満の話者を除外
- client_id を短い名前 (common_voice_N) にマッピング
- MP3 を WAV (22050Hz, mono, 16-bit) に変換
- 話者ごとに wav/ フォルダと transcripts_utf8.txt を作成
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from pydub import AudioSegment
from tqdm import tqdm


# 設定
DATASET_DIR = Path("g:/terashima/master/dataset/mcv-scripted-ja-v24.0")
OUTPUT_DIR = Path("g:/terashima/master/dataset/VITS_pretrain")
MIN_UTTERANCES = 10  # 最小発話数
SAMPLE_RATE = 22050  # サンプルレート (Hz)
CHANNELS = 1  # モノラル


def read_validated_tsv(tsv_path: Path) -> List[Dict[str, str]]:
    """
    validated.tsv を読み込み、データをリストとして返す。

    Args:
        tsv_path: TSV ファイルのパス

    Returns:
        各行のデータを辞書として含むリスト
    """
    print(f"TSV ファイルを読み込んでいます: {tsv_path}")
    data = []

    # CSV フィールドサイズの制限を最大値に設定
    import sys
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            data.append(row)

    print(f"総レコード数: {len(data)}")
    return data


def group_by_speaker(data: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    データを話者 (client_id) ごとにグループ化する。

    Args:
        data: TSV から読み込んだデータのリスト

    Returns:
        client_id をキーとし、その話者の発話リストを値とする辞書
    """
    print("話者ごとにグループ化しています...")
    grouped = defaultdict(list)
    
    for row in data:
        client_id = row.get('client_id', '')
        if client_id:
            grouped[client_id].append(row)
    
    print(f"総話者数: {len(grouped)}")
    return dict(grouped)


def filter_speakers(grouped_data: Dict[str, List[Dict[str, str]]], 
                   min_utterances: int) -> Dict[str, List[Dict[str, str]]]:
    """
    最小発話数に満たない話者を除外する。

    Args:
        grouped_data: 話者ごとにグループ化されたデータ
        min_utterances: 最小発話数

    Returns:
        フィルタリング後の話者データ
    """
    print(f"{min_utterances} 発話未満の話者を除外しています...")
    filtered = {
        client_id: utterances 
        for client_id, utterances in grouped_data.items() 
        if len(utterances) >= min_utterances
    }
    
    removed_count = len(grouped_data) - len(filtered)
    print(f"除外された話者数: {removed_count}")
    print(f"残りの話者数: {len(filtered)}")
    
    return filtered


def create_speaker_mapping(speaker_ids: List[str]) -> Dict[str, str]:
    """
    長い client_id を短い名前 (common_voice_1, common_voice_2, ...) にマッピングする。

    Args:
        speaker_ids: client_id のリスト

    Returns:
        client_id -> 短い名前 のマッピング辞書
    """
    print("話者IDマッピングを作成しています...")
    mapping = {}
    
    for idx, client_id in enumerate(sorted(speaker_ids), start=1):
        mapping[client_id] = f"common_voice_{idx}"
    
    return mapping


def save_speaker_mapping(mapping: Dict[str, str], output_path: Path):
    """
    話者IDマッピングを JSON ファイルとして保存する。

    Args:
        mapping: 話者IDマッピング辞書
        output_path: 出力ファイルパス
    """
    print(f"話者IDマッピングを保存しています: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, 
                       sample_rate: int = SAMPLE_RATE, 
                       channels: int = CHANNELS):
    """
    MP3 ファイルを WAV ファイルに変換する。

    Args:
        mp3_path: MP3 ファイルのパス
        wav_path: 出力 WAV ファイルのパス
        sample_rate: サンプルレート (Hz)
        channels: チャンネル数 (1=モノラル, 2=ステレオ)
    """
    # MP3 を読み込み
    audio = AudioSegment.from_mp3(mp3_path)
    
    # サンプルレートとチャンネル数を変更
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(channels)
    
    # WAV として保存 (16-bit PCM)
    audio.export(wav_path, format='wav', parameters=["-ac", str(channels)])


def process_speaker_data(client_id: str, 
                        utterances: List[Dict[str, str]], 
                        speaker_name: str,
                        clips_dir: Path,
                        output_dir: Path):
    """
    特定の話者のデータを処理する。

    Args:
        client_id: 話者の client_id
        utterances: その話者の発話リスト
        speaker_name: 短い話者名 (common_voice_N)
        clips_dir: MP3 ファイルが格納されているディレクトリ
        output_dir: 出力先ディレクトリ
    """
    # 話者ディレクトリと wav サブディレクトリを作成
    speaker_dir = output_dir / speaker_name
    wav_dir = speaker_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    # transcripts_utf8.txt 用のリスト
    transcripts = []
    
    # 各発話を処理
    for idx, utterance in enumerate(utterances, start=1):
        # ファイル名を取得
        mp3_filename = utterance.get('path', '')
        sentence = utterance.get('sentence', '')
        
        if not mp3_filename or not sentence:
            continue
        
        # MP3 ファイルのパス
        mp3_path = clips_dir / mp3_filename
        
        # MP3 ファイルが存在するか確認
        if not mp3_path.exists():
            print(f"警告: MP3 ファイルが見つかりません: {mp3_path}")
            continue
        
        # WAV ファイル名を生成 (common_voice_N_001.wav)
        wav_filename = f"{speaker_name}_{idx:03d}.wav"
        wav_path = wav_dir / wav_filename
        
        # MP3 を WAV に変換
        try:
            convert_mp3_to_wav(mp3_path, wav_path)
        except Exception as e:
            print(f"エラー: {mp3_path} の変換に失敗しました: {e}")
            continue
        
        # transcripts に追加 (拡張子なしのファイル名:transcript)
        wav_basename = wav_filename.replace('.wav', '')
        transcripts.append(f"{wav_basename}:{sentence}")
    
    # transcripts_utf8.txt を保存
    transcript_path = speaker_dir / "transcripts_utf8.txt"
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(transcripts))
    
    print(f"  → {speaker_name}: {len(transcripts)} ファイル処理完了")


def main():
    """
    メイン処理。
    """
    print("=" * 60)
    print("Common Voice データセット処理を開始します")
    print("=" * 60)
    
    # TSV ファイルのパス
    tsv_path = DATASET_DIR / "validated.tsv"
    clips_dir = DATASET_DIR / "clips"
    
    # TSV ファイルが存在するか確認
    if not tsv_path.exists():
        print(f"エラー: TSV ファイルが見つかりません: {tsv_path}")
        return
    
    # clips ディレクトリが存在するか確認
    if not clips_dir.exists():
        print(f"エラー: clips ディレクトリが見つかりません: {clips_dir}")
        return
    
    # Step 1: TSV を読み込み
    data = read_validated_tsv(tsv_path)
    
    # Step 2: 話者ごとにグループ化
    grouped_data = group_by_speaker(data)
    
    # Step 3: 話者をフィルタリング
    filtered_data = filter_speakers(grouped_data, MIN_UTTERANCES)
    
    # Step 4: 話者IDマッピングを作成
    speaker_ids = list(filtered_data.keys())
    speaker_mapping = create_speaker_mapping(speaker_ids)
    
    # Step 5: 出力ディレクトリを作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 6: 話者IDマッピングを保存
    mapping_path = OUTPUT_DIR / "speaker_mapping.json"
    save_speaker_mapping(speaker_mapping, mapping_path)
    
    # Step 7: 各話者のデータを処理
    print("\n各話者のデータを処理しています...")
    print("=" * 60)
    
    for client_id, utterances in tqdm(filtered_data.items(), desc="話者処理"):
        speaker_name = speaker_mapping[client_id]
        process_speaker_data(client_id, utterances, speaker_name, clips_dir, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("処理が完了しました！")
    print("=" * 60)
    print(f"出力先: {OUTPUT_DIR}")
    print(f"処理された話者数: {len(filtered_data)}")


if __name__ == "__main__":
    main()
