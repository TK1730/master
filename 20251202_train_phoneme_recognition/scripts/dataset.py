"""
音素認識用データセット読み込みスクリプト
"""
import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class PhonemeDataset(Dataset):
    """
    音素認識データを読み込むためのデータセットクラス
    リストファイルからファイルパスを読み込み、対応する _msp.npy と _ppgmat.npy ファイルをロードします。
    """
    def __init__(self, list_file, root_dir=None):
        """
        Args:
            list_file (str or Path): データファイルのリストを含むテキストファイルへのパス
            root_dir (str or Path, optional): ファイルパスの先頭に付加するルートディレクトリ
        """
        self.root_dir = root_dir
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
            msp = np.load(msp_path).astype(np.float32)
            ppg = np.load(ppg_path).astype(np.float32)

            # 必要に応じて (Channels, Time) に転置
            # 'in_channels' に基づきモデルが (Batch, Channels, Time) を期待していると仮定
            if msp.shape[1] == 80:  # (Time, 80) -> (80, Time)
                msp = msp.T
            if ppg.shape[1] == 36:  # (Time, 36) -> (36, Time)
                ppg = ppg.T

            return msp, ppg
        except Exception as e:
            print(f"Error loading {base_path}: {e}")
            raise e


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
    else:
        print(f"Train list not found at {train_list_path}")
