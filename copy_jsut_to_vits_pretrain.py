"""
JSUT Ver1.1 データセット統合スクリプト

このスクリプトは、jsut_ver1.1の複数のサブディレクトリから音声データと
トランスクリプトを読み取り、VITS_pretrainディレクトリに統合します。
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple


# 設定
SOURCE_BASE_DIR = Path(r"g:\terashima\master\dataset\jsut_ver1.1")
TARGET_BASE_DIR = Path(
    r"g:\terashima\master\dataset\VITS_pretrain\jsut_ver1.1-wav"
)

# 対象サブディレクトリ
SUBDIRS = [
    "basic5000",
    "countersuffix26",
    "loanword128",
    "onomatopee300",
    "precedent130",
    "repeat500",
    "travel1000",
    "utparaphrase512",
    "voiceactress100"
]


def read_transcript(transcript_path: Path) -> List[Tuple[str, str]]:
    """
    トランスクリプトファイルを読み込む
    
    Args:
        transcript_path: トランスクリプトファイルのパス
        
    Returns:
        (audio_id, text) のタプルのリスト
    """
    transcripts = []
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # "ID:text" 形式で分割
                if ':' in line:
                    parts = line.split(':', 1)
                    audio_id = parts[0].strip()
                    text = parts[1].strip() if len(parts) > 1 else ""
                    transcripts.append((audio_id, text))
                    
    except Exception as e:
        print(f"エラー: {transcript_path} の読み込みに失敗しました: {e}")
        
    return transcripts


def copy_audio_files(
    source_wav_dir: Path,
    target_wav_dir: Path,
    audio_ids: List[str]
) -> Tuple[int, int]:
    """
    音声ファイルをコピーする
    
    Args:
        source_wav_dir: 元の音声ディレクトリ
        target_wav_dir: コピー先のディレクトリ
        audio_ids: コピーする音声ファイルのIDリスト
        
    Returns:
        (成功数, スキップ数) のタプル
    """
    success_count = 0
    skipped_count = 0
    
    for audio_id in audio_ids:
        source_file = source_wav_dir / f"{audio_id}.wav"
        target_file = target_wav_dir / f"{audio_id}.wav"
        
        # 音声ファイルが存在する場合のみコピー
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                success_count += 1
            except Exception as e:
                print(f"警告: {source_file} のコピーに失敗しました: {e}")
                skipped_count += 1
        else:
            print(f"警告: {source_file} が見つかりません（スキップ）")
            skipped_count += 1
            
    return success_count, skipped_count


def main():
    """メイン処理"""
    print("=" * 60)
    print("JSUT Ver1.1 データセット統合スクリプト")
    print("=" * 60)
    print()
    
    # 出力ディレクトリの作成
    target_wav_dir = TARGET_BASE_DIR / "wav"
    target_wav_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリを作成しました: {target_wav_dir}")
    print()
    
    # 統合トランスクリプトの格納
    all_transcripts = []
    total_audio_files = 0
    total_success = 0
    total_skipped = 0
    
    # 各サブディレクトリを処理
    for subdir in SUBDIRS:
        print(f"処理中: {subdir}")
        print("-" * 60)
        
        source_dir = SOURCE_BASE_DIR / subdir
        transcript_path = source_dir / "transcript_utf8.txt"
        source_wav_dir = source_dir / "wav"
        
        # トランスクリプトファイルの存在確認
        if not transcript_path.exists():
            print(f"  警告: {transcript_path} が見つかりません")
            print()
            continue
            
        # トランスクリプトの読み込み
        transcripts = read_transcript(transcript_path)
        print(f"  トランスクリプト行数: {len(transcripts)}")
        
        if not transcripts:
            print(f"  警告: トランスクリプトが空です")
            print()
            continue
            
        # 音声ファイルのコピー
        audio_ids = [aid for aid, _ in transcripts]
        success, skipped = copy_audio_files(
            source_wav_dir,
            target_wav_dir,
            audio_ids
        )
        
        print(f"  コピー成功: {success} ファイル")
        print(f"  スキップ: {skipped} ファイル")
        print()
        
        # 統計を更新
        all_transcripts.extend(transcripts)
        total_audio_files += len(audio_ids)
        total_success += success
        total_skipped += skipped
    
    # 統合トランスクリプトファイルの作成
    output_transcript_path = TARGET_BASE_DIR / "transcription_utf8.txt"
    print(f"統合トランスクリプトファイルを作成中: {output_transcript_path}")
    
    try:
        with open(output_transcript_path, 'w', encoding='utf-8') as f:
            for audio_id, text in all_transcripts:
                f.write(f"{audio_id}:{text}\n")
        print(f"  統合トランスクリプト行数: {len(all_transcripts)}")
    except Exception as e:
        print(f"エラー: トランスクリプトファイルの作成に失敗しました: {e}")
        return
    
    # 最終結果の表示
    print()
    print("=" * 60)
    print("処理完了")
    print("=" * 60)
    print(f"総トランスクリプト行数: {len(all_transcripts)}")
    print(f"総音声ファイル数: {total_audio_files}")
    print(f"コピー成功: {total_success} ファイル")
    print(f"スキップ: {total_skipped} ファイル")
    print(f"出力先: {TARGET_BASE_DIR}")
    print()


if __name__ == "__main__":
    main()
