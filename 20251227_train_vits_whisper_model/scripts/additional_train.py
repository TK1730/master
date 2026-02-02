import csv
import json
import os
import random
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
# (scripts/ディレクトリから実行した際にmoduleをインポートできるようにする)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(project_root)

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import cv2
import numpy as np

from module.dataset_util import (
    AudioSpeakerTextLoader,
    collate_fn,
    slice_segments
)
from module.loss_function import (
    kl_divergence_loss,
    feature_loss,
    generator_adversarial_loss,
    discriminator_adversarial_loss
)
from module.vits_discriminator import VitsDiscriminator
from module.vits_generator import VitsGenerator


def main():
    load_iteration = "iteration50000"
    run(load_iteration=load_iteration, load_chekpoint=False)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    convert logscale
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def save_checkpoint(
    netG,
    netD,
    optimizerG,
    optimizerD,
    schedulerG,
    schedulerD,
    now_iteration,
    epoch,
    filepath
):
    checkpoint = {
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'schedulerG': schedulerG.state_dict(),
        'schedulerD': schedulerD.state_dict(),
        'now_iteration': now_iteration,
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath,
    netG,
    netD,
    optimizerG,
    optimizerD,
    schedulerG,
    schedulerD,
    device
):
    checkpoint = torch.load(filepath, map_location=device)
    netG.load_state_dict(checkpoint['netG'])
    netD.load_state_dict(checkpoint['netD'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    schedulerG.load_state_dict(checkpoint['schedulerG'])
    schedulerD.load_state_dict(checkpoint['schedulerD'])
    now_iteration = checkpoint.get('now_iteration', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {filepath}")
    return now_iteration, epoch


def load_train_dataset(train_dataset_txtfile_path, batch_size, manualSeed):
    # データセットを読み込む
    train_dataset = AudioSpeakerTextLoader(
        dataset_txtfile_path=train_dataset_txtfile_path,
    )

    # データローダーを生成
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=lambda worker_id: torch.manual_seed(
            manualSeed + worker_id
        )
    )
    print("train dataset size: {}".format(len(train_dataset)))
    return train_loader


def run(load_iteration="", load_chekpoint=False):
    # 乱数のシード設定
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 学習に必要なパラメーターをjsonファイルから読み込む
    config_path = "scripts/train_pseudo_whisper_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # 出力用ディレクトリがなければ作る
    output_dir = Path(config["save"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPUが使用可能かどうか確認
    device = torch.device(
        config["model"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # データセットを読み込む
    train_dataset_txtfile_path = config["data"]["train_dataset_txtfile_path"]
    batch_size = config["train"]["batch_size"]
    train_loader = load_train_dataset(
        train_dataset_txtfile_path,
        batch_size,
        manualSeed
    )
    test_dataset_txtfile_path = config["data"]["test_dataset_txtfile_path"]
    test_loader = load_train_dataset(
        test_dataset_txtfile_path,
        1,
        manualSeed
    )

    # Generatorのインスタンスを生成
    netG = VitsGenerator(
        n_phoneme=config["data"]["n_phoneme"],
        n_speakers=config["data"]["n_speakers"]
    )
    if load_chekpoint:
        pass
    netG = netG.to(device)

    # Discriminatorのインスタンスを生成
    netD = VitsDiscriminator()
    if load_chekpoint:
        pass
    netD = netD.to(device)

    # OptimizerをGeneratorとDiscriminatorに適用
    optimizerG = optim.AdamW(
        netG.parameters(),
        lr=config["train"]["lr"],
        betas=config["train"]["betas"],
        weight_decay=0.01
    )
    optimizerD = optim.AdamW(
        netD.parameters(),
        lr=config["train"]["lr"]/2.0,
        betas=config["train"]["betas"],
        weight_decay=0.01
    )

    # 学習率スケジューラの設定
    lr_schedulerG = optim.lr_scheduler.ExponentialLR(
        optimizerG,
        gamma=config["train"]["lr_decay"]
    )
    lr_schedulerD = optim.lr_scheduler.ExponentialLR(
        optimizerD,
        gamma=config["train"]["lr_decay"]
    )

    # GradScalerのインスタンスを生成
    scalerG = GradScaler()
    scalerD = GradScaler()

    now_iteration = 0
    epoch = 0
    if load_chekpoint:
        # チェックポイントからモデルとオプティマイザの状態を読み込む
        checkpoint_path = output_dir.joinpath(load_iteration, "checkpoint.pth")
        if checkpoint_path.exists():
            now_iteration, epoch = load_checkpoint(
                filepath=checkpoint_path,
                netG=netG,
                netD=netD,
                optimizerG=optimizerG,
                optimizerD=optimizerD,
                schedulerG=lr_schedulerG,
                schedulerD=lr_schedulerD,
                device=device
            )
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Checkpoint not found, starting from scratch.")

    # lossを記録することで学習過程を追うための変数
    # 学習が安定しているかをグラフから確認できるようにする
    losses_recorded = {
        "adversarial_loss/D": [],
        "adversarial_loss/G": [],
        "mel_reconstruction_loss/G": [],
        "kl_loss/G": [],
        "feature_matching_loss/G": []
    }

    kl_start_weight = 0.0
    kl_max_weight = 1.0
    kl_annealing_steps = 1  # どのくらいのステップでmaxにするか

    def get_kl_weight(epoch):
        return min(
            kl_max_weight,
            kl_start_weight + epoch / kl_annealing_steps * kl_max_weight
        )

    # tqdmプログレスバーを作成（total_iterationsまで）
    pbar = tqdm(
        total=config["train"]["total_iterations"],
        initial=now_iteration,
        desc="Training",
        unit="iter"
    )

    # 学習ループ: エポック数または総イテレーション数のいずれかの上限に達するまで継続
    while epoch <= config["train"]["epochs"]:
        epoch += 1
        kl_weight = get_kl_weight(epoch)
        now_iteration, should_continue = train_epoch(
            now_iteration=now_iteration,
            epoch=epoch,
            kl_weight=kl_weight,
            config=config,
            nets=(netG, netD),
            optims=(optimizerG, optimizerD),
            schedulers=(lr_schedulerG, lr_schedulerD),
            scaler=(scalerG, scalerD),
            loaders=(train_loader, test_loader),
            losses_recorded=losses_recorded,
            pbar=pbar,
            device=device
        )

        # イテレーション数が上限に達したら学習終了
        if not should_continue:
            pbar.close()
            print(f"学習終了: イテレーション数が上限 ({config['train']['total_iterations']})"
                  "に到達しました。")
            break

    pbar.close()


def compute_mel_from_wav(wav, fbanks, config, device):
    """
    波形からメルスペクトログラムを計算するヘルパー関数

    Args:
        wav: 入力波形 [batch, 1, time]
        fbanks: メルフィルターバンク
        config: 設定辞書
        device: デバイス

    Returns:
        mel_spec: メルスペクトログラム
    """
    # パディング
    pad_size = int(
        (config["data"]["filter_length"] - config["data"]["hop_length"]) / 2
    )
    wav_padded = torch.nn.functional.pad(
        wav,
        (pad_size, pad_size),
        mode='reflect'
    )

    # スペクトログラム計算
    spec = torchaudio.functional.spectrogram(
        waveform=wav_padded,
        pad=0,
        window=torch.hann_window(config["data"]["win_length"]).to(device),
        n_fft=config["data"]["filter_length"],
        hop_length=config["data"]["hop_length"],
        win_length=config["data"]["win_length"],
        power=1,
        normalized=False,
        center=False
    ).squeeze(1)

    # メルスペクトログラムへ変換
    mel_spec = torch.matmul(
        spec.clone().transpose(-1, -2),
        fbanks
    ).transpose(-1, -2)

    return mel_spec


def train_epoch(
    now_iteration,
    epoch,
    kl_weight,
    config,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    losses_recorded,
    pbar,
    device
):
    """
    1エポック分の学習を実行する関数
    Returns:
        now_iteration (int): 更新後のイテレーション数
        should_continue (bool): 学習を継続すべきかどうか (False=終了)
    """
    netG, netD = nets
    optimizerG, optimizerD = optims
    schedulerG, schedulerD = schedulers
    scalerG, scalerD = scaler
    train_loader, test_loader = loaders

    # データセットからbatch_size個ずつ取り出し学習
    for datas in train_loader:
        # 各データをdeviceに転送
        # collate_fnの返り値:
        # (wav_padded, wav_lengths, spec_padded, spec_lengths,
        #  speaker_id, mel_padded, mel_lengths)
        wav_real, _ = datas[0].to(device), datas[1].to(device)
        spec_real, spec_real_length = datas[2].to(device), datas[3].to(device)
        speaker_id = datas[4].to(device)
        # Whisperエンコーダーへの入力としてメルスペクトログラムを使用
        whisper_mel, whisper_mel_length = datas[5].to(device), datas[6].to(device)

        with autocast(device_type=device.type, dtype=torch.float16):
            # ===Generatorによる生成===
            (
                wav_fake,
                id_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = netG(
                whisper_mel, whisper_mel_length,
                spec_real, spec_real_length,
                speaker_id
            )

            # ===データセット中のスペクトログラムからメルスペクトログラムを計算===
            fbanks = torchaudio.functional.melscale_fbanks(
                n_freqs=config["data"]["filter_length"]//2 + 1,
                f_min=0,
                f_max=config["data"]["sampling_rate"]//2,
                n_mels=config["data"]["melspec_freq_dim"],
                sample_rate=config["data"]["sampling_rate"]).to(device)
            mel_spec_real = torch.matmul(
                spec_real.clone().transpose(-1, -2),
                fbanks
            ).transpose(-1, -2)
            mel_spec_real = dynamic_range_compression(mel_spec_real)
            # batch内の各メルスペクトログラム(上で計算したもの)について
            # id_sliceで指定されたindexから時間軸に沿って
            # (segment_size//hop_length)サンプル分取り出す
            # メルスペクトログラムをスライス
            segment_frames = (
                config["train"]["segment_size"] //
                config["data"]["hop_length"]
            )
            mel_spec_real = slice_segments(
                input_tensor=mel_spec_real,
                start_indices=id_slice,
                segment_size=segment_frames
            )

            # Generatorによって生成された波形からメルスペクトログラムを計算
            mel_spec_fake = compute_mel_from_wav(
                wav=wav_fake,
                fbanks=fbanks,
                config=config,
                device=device
            )
            mel_spec_fake = dynamic_range_compression(mel_spec_fake)

            # データセット中の波形「wav_real」について、batch内の各波形について、id_slice*hop_lengthで指定されたindexから時間軸に沿ってsegment_sizeサンプル分取り出す
            wav_real = slice_segments(
                input_tensor=wav_real,
                start_indices=id_slice*config["data"]["hop_length"],
                segment_size=config["train"]["segment_size"]
            )

            # ===Discriminatorの学習===
            # wav_real : 本物波形
            # wav_fake : 生成された波形
            authenticity_real, _ = netD(wav_real)
            authenticity_fake, _ = netD(wav_fake.detach())

            # lossを計算
            adversarial_loss_D, _, _ = discriminator_adversarial_loss(
                authenticity_real,
                authenticity_fake
            )  # adversarial loss

            # Discriminatorのlossの総計
            lossD = adversarial_loss_D

        # 勾配をリセット
        optimizerD.zero_grad()
        scalerD.scale(lossD).backward()

        # gradient explosionを避けるため勾配を制限
        scalerD.unscale_(optimizerD)
        nn.utils.clip_grad_norm_(
            netD.parameters(),
            max_norm=1.0,
            norm_type=2.0
        )

        scalerD.step(optimizerD)
        scalerD.update()

        with autocast(device_type=device.type, dtype=torch.float16):
            # ===Generatorの学習===
            authenticity_real, d_feature_map_real = netD(wav_real)
            authenticity_fake, d_feature_map_fake = netD(wav_fake)

            # ===lossを計算===
            # reconstruction loss
            mel_reconstruction_loss = F.l1_loss(
                mel_spec_fake,
                mel_spec_real
            ) * 45
            # KL divergence
            kl_loss = kl_divergence_loss(
                z_p,
                logs_q,
                m_p,
                logs_p,
                z_mask
            )
            kl_loss_weight = kl_loss * kl_weight
            feature_matching_loss = feature_loss(
                d_feature_map_real,
                d_feature_map_fake
            )
            # adversarial loss
            adversarial_loss_G, _ = generator_adversarial_loss(
                authenticity_fake
            )

            # Generatorのlossの総計
            lossG = (
                mel_reconstruction_loss +
                kl_loss_weight +
                feature_matching_loss +
                adversarial_loss_G
            )

        # 勾配をリセット
        optimizerG.zero_grad()
        # 勾配を計算
        scalerG.scale(lossG).backward()
        # gradient explosionを避けるため勾配を制限
        scalerG.unscale_(optimizerG)
        nn.utils.clip_grad_norm_(
            netG.parameters(),
            max_norm=1.0,
            norm_type=2.0
        )
        # パラメーターの更新
        scalerG.step(optimizerG)
        scalerG.update()

        # ===stdoutへlossを出力する===
        loss_stdout = {
            "adversarial_loss/D": adversarial_loss_D.item(),
            "adversarial_loss/G": adversarial_loss_G.item(),
            "mel_reconstruction_loss/G": mel_reconstruction_loss.item(),
            "kl_loss/G": min(kl_loss.item(), 200),
            "feature_matching_loss/G": feature_matching_loss.item()
        }
        if now_iteration % 100 == 0:
            print(
                f"[{now_iteration}/{config['train']['total_iterations']}]",
                end=""
            )
            for key, value in loss_stdout.items():
                # lossを記録
                losses_recorded[key].append(value)
                print(f" {key}:{value:.5f}", end="")
            print("")

            # mel_spec_realとmel_spec_fakeをOpenCVでリアルタイム表示
            mel_real_np = mel_spec_real[0].detach().cpu().numpy()
            mel_fake_np = mel_spec_fake[0].detach().cpu().numpy()

            # numpyで0-255の範囲に正規化
            mel_real_min = mel_real_np.min()
            mel_real_max = mel_real_np.max()
            mel_real_norm = (
                (mel_real_np - mel_real_min) /
                (mel_real_max - mel_real_min + 1e-8) * 255
            ).astype(np.uint8)

            mel_fake_min = mel_fake_np.min()
            mel_fake_max = mel_fake_np.max()
            mel_fake_norm = (
                (mel_fake_np - mel_fake_min) /
                (mel_fake_max - mel_fake_min + 1e-8) * 255
            ).astype(np.uint8)

            # 上下反転
            mel_real_color = cv2.flip(mel_real_norm, 0)
            mel_fake_color = cv2.flip(mel_fake_norm, 0)

            # リサイズして見やすく
            height = 200
            real_width = int(height * mel_real_color.shape[1] / mel_real_color.shape[0])
            fake_width = int(height * mel_fake_color.shape[1] / mel_fake_color.shape[0])

            mel_real_resized = cv2.resize(mel_real_color, (real_width, height))
            mel_fake_resized = cv2.resize(mel_fake_color, (fake_width, height))

            # 幅を揃える
            max_w = max(real_width, fake_width)
            if real_width < max_w:
                pad = max_w - real_width
                mel_real_resized = cv2.copyMakeBorder(
                    mel_real_resized, 0, 0, 0, pad,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            if fake_width < max_w:
                pad = max_w - fake_width
                mel_fake_resized = cv2.copyMakeBorder(
                    mel_fake_resized, 0, 0, 0, pad,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            # 結合して表示
            combined = cv2.vconcat([mel_real_resized, mel_fake_resized])
            cv2.imshow('Mel Spectrograms', combined)
            cv2.waitKey(1)

        # ===学習状況をファイルに出力===
        if (
            now_iteration % config["train"]["output_iter"] == 0 and
            now_iteration != 0
        ):
            out_dir = os.path.join(
                config["save"]["output_dir"],
                f"iteration{now_iteration}"
            )
            # 出力用ディレクトリがなければ作る
            os.makedirs(out_dir, exist_ok=True)

            # ===チェックポイントを保存===
            save_checkpoint(
                netG=netG,
                netD=netD,
                optimizerG=optimizerG,
                optimizerD=optimizerD,
                schedulerG=schedulerG,
                schedulerD=schedulerD,
                now_iteration=now_iteration,
                epoch=epoch,
                filepath=os.path.join(out_dir, "checkpoint.pth")
            )

            # ===評価を実行（音声生成とメルスペクトログラム損失計算）===
            eval_output_path = os.path.join(out_dir, "eval_samples")
            eval_losses = evaluate(
                nets=(netG, netD),
                loader=test_loader,
                config=config,
                device=device,
                output_path=eval_output_path
            )

            # ===学習済みモデル（CPU向け）を出力===
            # generatorを出力
            netG.eval()
            torch.save(
                netG.to('cpu').state_dict(),
                os.path.join(out_dir, "netG_cpu.pth")
            )
            netG.to(device)
            netG.train()
            # discriminatorを出力
            netD.eval()
            torch.save(
                netD.to('cpu').state_dict(),
                os.path.join(out_dir, "netD_cpu.pth")
            )
            netD.to(device)
            netD.train()

            # lossのグラフを出力
            plt.clf()
            plt.figure(figsize=(16, 6))
            plt.subplots_adjust(wspace=0.4, hspace=0.6)
            for i, (loss_name, loss_list) in enumerate(
                losses_recorded.items()
            ):
                plt.subplot(2, 3, i+1)
                plt.title(loss_name)
                plt.plot(loss_list, label="loss")
                plt.xlabel("iterations")
                plt.ylabel("loss")
                plt.legend()
                plt.grid()
            plt.savefig(os.path.join(out_dir, "loss.png"))
            plt.close()

            # ===CSVにlossを保存===
            # 現在のイテレーションの平均train loss
            current_train_losses = {
                key: sum(value[-100:]) / min(len(value), 100)
                for key, value in losses_recorded.items()
            }

            csv_path = os.path.join(
                config["save"]["output_dir"],
                "loss_history.csv"
            )
            # CSVファイルが存在しない場合はヘッダーを書き込む
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'iteration',
                    'epoch',
                    'train_adversarial_loss_D',
                    'train_adversarial_loss_G',
                    'train_mel_reconstruction_loss_G',
                    'train_kl_loss_G',
                    'train_feature_matching_loss_G',
                    'eval_mel_l1_loss'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'iteration': now_iteration,
                    'epoch': epoch,
                    'train_adversarial_loss_D': current_train_losses.get(
                        'adversarial_loss/D', 0.0
                    ),
                    'train_adversarial_loss_G': current_train_losses.get(
                        'adversarial_loss/G', 0.0
                    ),
                    'train_mel_reconstruction_loss_G': (
                        current_train_losses.get(
                            'mel_reconstruction_loss/G', 0.0
                        )
                    ),
                    'train_kl_loss_G': current_train_losses.get(
                        'kl_loss/G', 0.0
                    ),
                    'train_feature_matching_loss_G': (
                        current_train_losses.get(
                            'feature_matching_loss/G', 0.0
                        )
                    ),
                    'eval_mel_l1_loss': eval_losses['mel_l1_loss']
                })
            print(f"Loss history saved to {csv_path}")

        now_iteration += 1
        pbar.update(1)  # プログレスバーを1進める

        # イテレーション数が上限に達したらエポック途中でも終了
        if now_iteration >= config["train"]["total_iterations"]:
            return now_iteration, False  # 学習終了を通知

    # 学習率を更新 (エポック終了時)
    schedulerG.step()
    schedulerD.step()

    # エポック完了、学習は継続
    return now_iteration, True


def evaluate(nets, loader, config, device, output_path="output/eval"):
    """
    評価関数: netGを使って音声を生成し保存する

    Args:
        nets: (netG, netD) のタプル
        loader: データローダー
        config: 設定辞書
        device: デバイス
        output_path: 出力ディレクトリのパス

    Returns:
        eval_losses: 評価時の各種損失を含む辞書
    """
    netG, netD = nets
    netG.eval()

    # 出力ディレクトリを作成
    os.makedirs(output_path, exist_ok=True)

    # 評価用の損失を記録するリスト
    mel_l1_losses = []

    with torch.no_grad():
        # 最初の3バッチのみを評価に使用
        for batch_idx, datas in enumerate(loader):
            if batch_idx >= 3:  # 最初の3バッチのみ処理
                break

            # データをdeviceに転送
            wav_real, _ = datas[0].to(device), datas[1].to(device)
            spec_real, spec_real_length = (
                datas[2].to(device), datas[3].to(device)
            )
            speaker_id = datas[4].to(device)
            whisper_mel, whisper_mel_length = (
                datas[5].to(device), datas[6].to(device)
            )

            # netGで音声を生成
            wav_fake = netG.text_to_speech(
                whisper_mel,
                spec_real,
                spec_real_length,
                speaker_id
            )

            # メルスペクトログラムの計算とL1損失の算出
            fbanks = torchaudio.functional.melscale_fbanks(
                n_freqs=config["data"]["filter_length"]//2 + 1,
                f_min=0,
                f_max=config["data"]["sampling_rate"]//2,
                n_mels=config["data"]["melspec_freq_dim"],
                sample_rate=config["data"]["sampling_rate"]
            ).to(device)

            # 正解メルスペクトログラムを計算
            mel_spec_real = torch.matmul(
                spec_real.clone().transpose(-1, -2),
                fbanks
            ).transpose(-1, -2)
            mel_spec_real = dynamic_range_compression(mel_spec_real)

            # 生成されたメルスペクトログラムを計算
            mel_spec_fake = compute_mel_from_wav(
                wav=wav_fake,
                fbanks=fbanks,
                config=config,
                device=device
            )
            mel_spec_fake = dynamic_range_compression(mel_spec_fake)

            # L1損失を計算
            mel_l1_loss = F.l1_loss(mel_spec_fake, mel_spec_real)
            mel_l1_losses.append(mel_l1_loss.item())

            # バッチ内の最初の3サンプルを保存
            num_samples = min(3, wav_fake.shape[0])
            for i in range(num_samples):
                # 生成された音声を保存
                output_filename = os.path.join(
                    output_path,
                    f"batch{batch_idx}_sample{i}_generated.wav"
                )
                torchaudio.save(
                    output_filename,
                    wav_fake[i].cpu(),
                    config["data"]["sampling_rate"]
                )

                # 元の音声も保存（比較用）
                output_filename_real = os.path.join(
                    output_path,
                    f"batch{batch_idx}_sample{i}_real.wav"
                )
                torchaudio.save(
                    output_filename_real,
                    wav_real[i].cpu(),
                    config["data"]["sampling_rate"]
                )

    # 平均L1損失を計算
    avg_mel_l1_loss = sum(mel_l1_losses) / len(mel_l1_losses)

    netG.train()
    print(f"評価完了: 生成された音声を {output_path} に保存しました")
    print(f"評価メルスペクトログラムL1損失: {avg_mel_l1_loss:.5f}")

    return {
        "mel_l1_loss": avg_mel_l1_loss
    }


if __name__ == "__main__":
    main()
