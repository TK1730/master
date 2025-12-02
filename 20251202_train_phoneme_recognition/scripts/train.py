"""
音素認識モデルの訓練スクリプト
"""
from types import SimpleNamespace
from typing import Union
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import os

from model.lstm_net_rev import LSTM_net


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


def draw_msp(y, t):
    """
    MSPを表示する
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
    heatmap[ans, pre] += 1
    heatmap = cv2.resize(heatmap, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("h-map", heatmap)
    cv2.waitKey(1)


def create_config() -> SimpleNamespace:
    """
    設定値を作成する関数

    Returns:
        SimpleNamespace: 設定値
    """
    config = SimpleNamespace()
    config.hyper_params = SimpleNamespace(
        epochs=3000,
        frame_length=88,
        batch_size=128,
        test_rate=0.2,
        learning_rate=2e-4,
        train_decay=0.998,
        eps=1.0e-9,
        seed=1234,
        input_type=['msp'],
        output_type=['ppgmat'],
        jvs=True,
        pseudo=True,
        whisper=True,
        special=False
    )
    config.model_params = SimpleNamespace(
        in_channels=80,
        hidden_channels=256,
        out_channels=36,  # 音素数
        kernel_size=3,
        dilation_rate=2,
        n_layers=4,
        p_dropout=0.1,
        layernorm=True,
        activation="gate",
        bidirectional=True,
        l2softmax=False,
        continuous=False,
        fc_size=128
    )
    return config


def softmax(x, axis=-1):
    """
    Softmax関数
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-axis, keepdims=True)


def create_model(config: dict, load_checkpoint: dict = None) -> nn.Module:
    """LSTMモデルの作成

    Args:
        config (dict): 設定値
        load_checkpoint (dict, optional): 読み込むチェックポイント. Defaults to None.

    Returns:
        nn.Module: LSTMモデル
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
    # config設定作成
    config = create_config()
    # 乱数の設定
    set_seed(config.hyper_params.seed)
    # デバイスの設定
    device = use_device()
    # モデルの作成
    model = create_model(config.model_params.__dict__)
    model.to(device)
    # データローダーの作成
    train_dataset = ""
    test_dataset = ""
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)
    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()
    # オプティマイザーの設定
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # ログマネージャー
    log_manager = LogManager()
    save_folder = "result"
    os.makedirs(save_folder, exist_ok=True)

    logs = []
    min_valid_loss = 100000.0

    # 学習
    for epoch in range(1, config.hyper_params.epochs + 1):
        train_loss, val_loss, train_correct, val_correct, y, t, pre, ans = train_and_evaluate(
            device, model, train_loader, test_loader, criterion, optimizer
        )

        # 画像表示
        draw_msp(y, t)
        draw_heatmap(pre, ans)

        # ベストモデルの保存
        if val_loss < min_valid_loss:
            log_manager.save_model(model, "model", save_folder)
            min_valid_loss = val_loss

        # ログ保存
        log_epoch = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'min_val_loss': min_valid_loss, 
            'train_correct': train_correct,
            'val_correct': val_correct,
        }
        logs.append(log_epoch)
        log_manager.save_logs(logs, save_folder)

        # チェックポイント保存
        checkpoint_e = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": val_loss # 修正: loss変数はループ外では最後のバッチのlossになるため、val_lossを使用するか、適切に渡す必要があるが、ここではval_lossで代用
        }
        log_manager.save_checkpoint(checkpoint_e, save_folder, file_name="encoder_checkpoint.pth")

        print(f"epoch: {epoch} train_loss:{train_loss:.6f} val_loss: {val_loss:.6f} min_val_loss: {min_valid_loss}")
        print(f"train_correct: {train_correct:.2f} test_correct: {val_correct:.2f}")


def use_device() -> str:
    """使用するデバイスを返す

    Returns:
        str: 使用するデバイス
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return str(device)


def set_seed(seed: int):
    """乱数の再現性を保つために乱数の種を設定する

    Args:
        seed (int): 乱数の種
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train_and_evaluate(device, model, train_loader, test_loader, criterion, optim):
    """訓練と検証を行う

    Args:
        device (torch.device): デバイス
        model (nn.Module): モデル
        train_loader (DataLoader): 訓練データローダー
        test_loader (DataLoader): テストデータローダー
        criterion (nn.Module): 損失関数
        optim (nn.Module): オプティマイザー
    """
    # 反精度学習
    scaler = torch.amp.GradScaler()

    epoch_train_loss, epoch_val_loss = 0.0, 0.0
    epoch_train_corrects, epoch_val_corrects = 0, 0

    # 戻り値用の変数初期化
    y_out, t_out, pre_out, ans_out = None, None, None, None

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = test_loader
        
        for j, (v, t) in enumerate(loader):
            # 勾配の初期化
            optim.zero_grad()
            v, t = v.to(device), t.to(device)
            with torch.set_grad_enabled(phase == 'train'):
                with torch.amp.autocast(device_type=str(device), dtype=torch.bfloat16):
                    y = model(v)
                    # loss
                    loss = criterion(y, t)
                    pre = torch.argmax(y, dim=1)
                    ans = torch.argmax(t, dim=1)

                    if phase == 'train':
                        scaler.scale(loss).backward()

                        # optimizerの更新
                        scaler.step(optim)
                        scaler.update()
                    
                        # ロスの集計
                        epoch_train_loss += loss.item() / len(loader)
                        # 正解率
                        epoch_train_corrects += torch.sum(pre == ans)
                    
                    else:
                        epoch_val_loss += loss.item() / len(loader)
                        # 正解率
                        epoch_val_corrects += torch.sum(pre == ans)

                        # 可視化用に最後のバッチのデータを保存
                        if j == len(loader) - 1:
                            y_out = y
                            t_out = t
                            pre_out = pre
                            ans_out = ans

    # epochごとの正解率 (frame_lengthはconfigから取得するか、引数で渡す必要があるが、ここでは簡易的に計算)
    # 注意: frame_lengthが未定義のため、一旦コメントアウトまたは修正が必要。
    # 元のコードではグローバル変数的に参照していた可能性があるが、ここではloaderのdatasetサイズで割る形にする。
    # ただし、frame_length倍されている意図が不明確なため、単純な平均正解率にするか、元のロジックを尊重するか。
    # ここでは単純にサンプル数で割る形に修正（frame単位の正解率なら総フレーム数で割るべき）
    
    # 簡易的な修正: datasetの長さ * 88 (frame_length) と仮定
    frame_length = 88 
    epoch_train_corrects = epoch_train_corrects.double() / (len(train_loader.dataset) * frame_length)
    epoch_val_corrects = epoch_val_corrects.double() / (len(test_loader.dataset) * frame_length)
        
    return epoch_train_loss, epoch_val_loss, epoch_train_corrects, epoch_val_corrects, y_out, t_out, pre_out, ans_out


if __name__ == '__main__':
    main()
