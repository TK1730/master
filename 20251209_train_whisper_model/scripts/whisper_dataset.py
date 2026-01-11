from pathlib import Path
import pandas as pd
import librosa
from typing import Dict, Any, Union
from datasets import Dataset


def load_audio_from_path(
    batch: Dict[str, Any]
) -> Dict[str, Any]:
    """
    データセットのマッピング用関数。音声ファイルをロードします。

    Args:
        batch: データセットのバッチ(1行分のデータ)

    Returns:
        音声データとテキストを含む辞書

        {'audio': {'path': path, 'array': audio, 'sampling_rate': 16000},
        'labels': labels}
    """
    path = batch["audio_path"]
    transcription = batch["transcription"]

    # 音声のロード (16kHz)
    try:
        audio, _ = librosa.load(path, sr=16000)
    except Exception as e:
        print(f"Warning: Failed to load audio file {path}: {e}")
        raise e

    return {
        "audio": {
            "path": path,
            "array": audio,
            "sampling_rate": 16000
        },
        "labels": transcription
    }


def create_dataset(
    train_csv: Union[str, Path],
    test_csv: Union[str, Path],
    processor: Any,
) -> Any:
    """
    CSVファイルからWhisper学習用のDatasetオブジェクトを作成します。

    Args:
        train_csv: 学習用CSVファイルのパス
        test_csv: テスト用CSVファイルのパス

    Returns:
        (train_dataset, test_dataset): 処理済みのDatasetオブジェクトのタプル
    """

    print(f"Loading datasets from:\n  Train: {train_csv}\n  Test: {test_csv}")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 1. 音声データの読み込み
    print("Loading audio data for train dataset...")
    train_dataset = train_dataset.map(
        load_audio_from_path,
        remove_columns=train_dataset.column_names,

    )

    print("Loading audio data for test dataset...")
    test_dataset = test_dataset.map(
        load_audio_from_path,
        remove_columns=test_dataset.column_names,

    )

    # 2. 特徴量とラベルの作成
    print("Preparing features and labels for train dataset...")
    train_dataset = train_dataset.map(
        prepare_dataset,
        fn_kwargs={
            "feature_extractor": processor.feature_extractor,
            "tokenizer": processor.tokenizer
        },

    )

    print("Preparing features and labels for test dataset...")
    test_dataset = test_dataset.map(
        prepare_dataset,
        fn_kwargs={
            "feature_extractor": processor.feature_extractor,
            "tokenizer": processor.tokenizer
        },

    )

    return train_dataset, test_dataset


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(
        batch["labels"]
    ).input_ids

    return batch


if __name__ == "__main__":
    import os

    train_csv = 'dataset/train.csv'
    test_csv = 'dataset/test.csv'

    # Create debug CSVs with just 5 rows to test the pipeline quickly
    debug_train_csv = 'dataset/debug_train.csv'
    debug_test_csv = 'dataset/debug_test.csv'

    if os.path.exists(train_csv):
        pd.read_csv(train_csv).head(5).to_csv(debug_train_csv, index=False)
    else:
        print(f"Warning: {train_csv} not found.")

    if os.path.exists(test_csv):
        pd.read_csv(test_csv).head(5).to_csv(debug_test_csv, index=False)
    else:
        print(f"Warning: {test_csv} not found.")

    if os.path.exists(debug_train_csv) and os.path.exists(debug_test_csv):
        print("Running create_dataset with debug CSVs...")

        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-small",
            language="Japanese",
            task="transcribe"
        )

        train_ds, test_ds = create_dataset(
            debug_train_csv, debug_test_csv, processor=processor
        )

        print("Verification Results (Train)")
        if len(train_ds) > 0:
            sample = train_ds[0]
            print("Sample keys:", sample.keys())

            if "input_features" in sample:
                feat = sample['input_features']
                if len(feat) > 0:
                    shape_str = f"{len(feat)}x{len(feat[0])}"
                else:
                    shape_str = "0"
                print(f"Input features shape: {shape_str}")

            if "labels" in sample:
                print(f"Labels length: {len(sample['labels'])}")
                print(f"Labels (first 10): {sample['labels'][:10]}")
        else:
            print("Train dataset is empty.")

    # helper for verification (no cleanup to allow inspection if needed)
    print("Done.")
