"""
音素認識用データセット読み込みスクリプト
"""
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PhonemeDataset(Dataset):
    """
    音素認識データを読み込むためのデータセットクラス
    リストファイルからファイルパスを読み込み、対応する _msp.npy と _ppgmat.npy ファイルをロードします。
    """
    def __init__(
        self,
        list_file: Union[str, Path],
        root_dir: Union[str, Path] = None,
        frame_length: int = 88,
    ):
        """
        Args:
            list_file (str or Path): データファイルのリストを含むテキストファイルへのパス
            root_dir (str or Path, optional): ファイルパスの先頭に付加するルートディレクトリ
            frame_length (int, optional): フレーム長
        """
        self.root_dir = root_dir
        self.frame_length = frame_length
        with open(list_file, 'r', encoding='utf-8') as f:
            self.files = [line.strip() for line in f]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base_path = self.files[idx]
        if self.root_dir:
            file_path = os.path.join(self.root_dir, base_path)
        else:
            file_path = base_path

        msp_path = file_path + "_msp.npy"
        ppg_path = file_path + "_ppgmat.npy"

        try:
            msp = np.load(msp_path).astype(np.float32)  # -> time, freq
            ppg = np.load(ppg_path).astype(np.float32)  # -> time, freq

            start_idx = np.random.randint(0, msp.shape[0] - self.frame_length)
            msp = msp[start_idx:start_idx + self.frame_length]
            ppg = ppg[start_idx:start_idx + self.frame_length]

            return torch.from_numpy(msp), torch.from_numpy(ppg)
        except Exception as e:
            print(f"Error loading {base_path}: {e}")
            raise e


def collate_fn(batch):
    """
    バッチ内のシーケンスをパディングする関数
    """
    msp_batch, ppg_batch = zip(*batch)

    # パディング (Batch, Time, Channels)
    msp_padded = pad_sequence(msp_batch, batch_first=True)
    ppg_padded = pad_sequence(ppg_batch, batch_first=True)

    return msp_padded, ppg_padded


if __name__ == '__main__':
    # 検証ブロック
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    train_list_path = dataset_dir / "train.txt"

    if train_list_path.exists():
        print(f"Loading dataset from {train_list_path}")
        dataset = PhonemeDataset(train_list_path, root_dir=dataset_dir)
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            msp, ppg = dataset[0]
            print(f"Sample 0 MSP shape: {msp.shape}")
            print(f"Sample 0 PPG shape: {ppg.shape}")

            # Collate check
            batch = [dataset[0], dataset[1]]
            msp_pad, ppg_pad = collate_fn(batch)
            print(f"Batch MSP shape: {msp_pad.shape}")
            print(f"Batch PPG shape: {ppg_pad.shape}")
    else:
        print(f"Train list not found at {train_list_path}")
