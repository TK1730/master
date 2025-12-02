"""
音素認識モデルの訓練スクリプト
"""
from types import SimpleNamespace
from typing import Tuple
import cv2
import numpy as np
from pathlib import Path
import yaml
import sys
import datetime
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.lstm_net_rev import LSTM_net  # noqa: E402
from scripts.dataset import PhonemeDataset, collate_fn  # noqa: E402


class LogManager:
    """
    学習ログを管理するクラス
    """
    def __init__(self):
        pass

    def save_model(self, model, name, folder):
        """
        モデルを保存する
        """
        torch.save(model.state_dict(), f"{folder}/{name}.pth")

    def save_logs(self, logs, folder):
        """
        ログを保存する
        """
        df = pd.DataFrame(logs)
        df.to_csv(f"{folder}/log.csv", index=False)

    def save_checkpoint(self, checkpoint, folder, file_name):
        """
        チェックポイントを保存する
        """
        torch.save(checkpoint, f"{folder}/{file_name}")

    def save_config(self, config, folder):
        """
        設定を保存する
        """
        with open(f"{folder}/config.yaml", 'w') as f:
            yaml.dump(config, f)


def draw_msp(y, t):
    """
    ppgを表示する
    """
    y = y.to(torch.float32).to('cpu').detach().numpy()[0]
    t = t.to(torch.float32).to('cpu').detach().numpy()[0]
    y = softmax(y, axis=0)
    img = np.vstack((t, y))
    img = cv2.resize(
        img,
        dsize=None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_LINEAR
    )
    cv2.imshow("ppg", img)
    cv2.waitKey(1)


def draw_heatmap(pre, ans):
    """
    ヒートマップを表示する
    """
    ans = ans.to(torch.int32).to('cpu').detach().numpy()
    pre = pre.to(torch.int32).to('cpu').detach().numpy()
    heatmap = np.zeros((36, 36), dtype=np.float32)

    ans = ans.flatten()
    pre = pre.flatten()

    mask = (ans < 36) & (pre < 36)
    ans = ans[mask]
    pre = pre[mask]

    np.add.at(heatmap, (ans, pre), 1)

    heatmap = heatmap / (heatmap.sum(axis=1, keepdims=True) + 1e-9)

    heatmap = cv2.resize(
        heatmap,
        dsize=None,
        fx=10,
        fy=10,
        interpolation=cv2.INTER_NEAREST
    )
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imshow("h-map", heatmap)
    cv2.waitKey(1)


def create_config(config_path="config.yaml") -> SimpleNamespace:
    """
    設定値を作成する関数
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace()
    config.hyper_params = SimpleNamespace(**config_dict['hyperparameters'])
    config.model_params = SimpleNamespace(**config_dict['model_params'])
    config.sampling_rate = config_dict.get('sampling_rate', 22050)
    config.model_name = config_dict.get('model_name', 'LSTM_net')

    return config


def softmax(x, axis=-1):
    """
    Softmax関数
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-axis, keepdims=True)


def create_model(config: dict, load_checkpoint: str = None) -> nn.Module:
    """LSTMモデルの作成
    """
    lstm_net = LSTM_net(
        n_inputs=config["in_channels"],
        n_outputs=config["out_channels"],
        n_layers=config["n_layers"],
        hidden_size=config["hidden_channels"],
        fc_size=config["fc_size"],
        dropout=config["p_dropout"],
        bidirectional=config["bidirectional"],
        l2softmax=config["l2softmax"],
        continuous=config["continuous"]
    )
    if load_checkpoint is not None:
        lstm_net.load_state_dict(torch.load(load_checkpoint))
    return lstm_net


def main():
    """
    音素認識モデル学習
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"

    config = create_config(str(config_path))

    set_seed(config.hyper_params.seed)
    device = use_device()

    model = create_model(config.model_params.__dict__)
    model.to(device)

    dataset_dir = project_root / "dataset"
    train_dataset = PhonemeDataset(
        list_file=dataset_dir / "train.txt",
        root_dir=dataset_dir
    )
    test_dataset = PhonemeDataset(
        list_file=dataset_dir / "test.txt",
        root_dir=dataset_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.hyper_params.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.hyper_params.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=float(config.hyper_params.learning_rate)
    )

    scaler = GradScaler()

    log_manager = LogManager()

    experiment_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_folder = project_root / "trained_models" / experiment_id
    os.makedirs(save_folder, exist_ok=True)

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    log_manager.save_config(config_dict, save_folder)

    logs = []
    min_valid_loss = 100000.0

    print(f"Start training... Experiment ID: {experiment_id}")
    print(f"Train samples: {len(train_dataset)}, "
          f"Test samples: {len(test_dataset)}")

    for epoch in range(1, config.hyper_params.epochs + 1):
        (train_loss, val_loss, train_correct,
         val_correct, y, t, pre, ans) = train_and_evaluate(
            device, model, train_loader, test_loader, criterion,
            optimizer, scaler
        )

        if y is not None:
            draw_msp(y, t)
            draw_heatmap(pre, ans)

        if val_loss < min_valid_loss:
            log_manager.save_model(model, "model", save_folder)
            min_valid_loss = val_loss
            print(f"New best model saved at epoch {epoch} "
                  f"(val_loss: {val_loss:.6f})")

        log_epoch = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'min_val_loss': min_valid_loss,
            'train_correct': train_correct.item(),
            'val_correct': val_correct.item(),
        }
        logs.append(log_epoch)
        log_manager.save_logs(logs, save_folder)

        checkpoint_e = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss
        }
        log_manager.save_checkpoint(
            checkpoint_e,
            save_folder,
            file_name="checkpoint.pth"
        )

        print(f"epoch: {epoch}  "
              f"train_loss:{train_loss:.6f} "
              f"val_loss: {val_loss:.6f} "
              f"min_val_loss: {min_valid_loss:.6f}")

        print(f"train_correct: {train_correct:.4f} "
              f"test_correct: {val_correct:.4f}")


def use_device() -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return str(device)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train_and_evaluate(
    device: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optim: nn.Module,
    scaler: GradScaler,
) -> Tuple[float, float, int, int, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:

    epoch_train_loss, epoch_val_loss = 0.0, 0.0
    epoch_train_corrects, epoch_val_corrects = 0, 0
    total_train_frames, total_val_frames = 0, 0

    y_out, t_out, pre_out, ans_out = None, None, None, None

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = test_loader

        for j, (v, t) in enumerate(loader):
            optim.zero_grad()
            v, t = v.to(device), t.to(device)

            with torch.set_grad_enabled(phase == 'train'):
                with autocast(device_type=device, dtype=torch.bfloat16):
                    y = model(v)

                    # Permute to (Batch, Channels, Time) for loss calculation
                    # y: (Batch, Time, 36) -> (Batch, 36, Time)
                    y = y.permute(0, 2, 1)

                    # t: (Batch, Time, 36) -> (Batch, 36, Time)
                    t = t.permute(0, 2, 1)

                    # Target for CrossEntropyLoss should be indices
                    # (Batch, Time)
                    t_indices = torch.argmax(t, dim=1)

                    loss = criterion(y, t_indices)

                    pre = torch.argmax(y, dim=1)
                    ans = t_indices

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()

                        epoch_train_loss += loss.item() * v.size(0)
                        epoch_train_corrects += torch.sum(pre == ans)
                        total_train_frames += (ans.size(0) * ans.size(1))

                    else:
                        epoch_val_loss += loss.item() * v.size(0)
                        epoch_val_corrects += torch.sum(pre == ans)
                        total_val_frames += (ans.size(0) * ans.size(1))

                        if j == len(loader) - 1:
                            y_out = y
                            t_out = t
                            pre_out = pre
                            ans_out = ans

    if len(train_loader.dataset) > 0:
        epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
    if len(test_loader.dataset) > 0:
        epoch_val_loss = epoch_val_loss / len(test_loader.dataset)

    if total_train_frames > 0:
        epoch_train_corrects = (
            epoch_train_corrects.double() / total_train_frames
        )
    if total_val_frames > 0:
        epoch_val_corrects = epoch_val_corrects.double() / total_val_frames

    return (
        epoch_train_loss,
        epoch_val_loss,
        epoch_train_corrects,
        epoch_val_corrects,
        y_out,
        t_out,
        pre_out,
        ans_out
    )


if __name__ == '__main__':
    main()
