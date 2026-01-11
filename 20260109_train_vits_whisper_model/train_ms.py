import cv2
import os
import tqdm
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import gc

from pathlib import Path
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from typing import List

import utils.task as task
from utils.logger import logger
from data_utils import *
from module.vits2_generator import Vits2Generator
from module.vits2_discriminator import MultiPeriodDiscriminator, DurationDiscriminator
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    kl_loss_normal,
)
from utils.mel_processing import (spec_to_mel, wav_to_mel)
from utils.model import slice_segments, clip_grad_value_
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from utils.hyper_parameters import HyperParameters
from nlp.symbols import SYMBOLS

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    load_iteration = "iteration29000"
    run(load_iteration=load_iteration, load_chekpoint=True)


def save_checkpoint(
    netG,
    netD,
    netD_dur,
    optimizerG,
    optimizerD,
    optimizerD_dur,
    schedulerG,
    schedulerD,
    schedulerD_dur,
    now_iteration,
    epoch,
    filepath,
):
    checkpoint = {
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "netD_dur": netD_dur.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "optimizerD_dur": optimizerD_dur.state_dict(),
        "schedulerG": schedulerG.state_dict(),
        "schedulerD": schedulerD.state_dict(),
        "schedulerD_dur": schedulerD_dur.state_dict(),
        "now_iteration": now_iteration,
        "epoch": epoch,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath,
    netG,
    netD,
    netD_dur,
    optimizerG,
    optimizerD,
    optimizerD_dur,
    schedulerG,
    schedulerD,
    schedulerD_dur,
    device,
):
    checkpoint = torch.load(filepath, map_location=device)
    netG.load_state_dict(checkpoint["netG"])
    netD.load_state_dict(checkpoint["netD"])
    netD_dur.load_state_dict(checkpoint["netD_dur"])
    optimizerG.load_state_dict(checkpoint["optimizerG"])
    optimizerD.load_state_dict(checkpoint["optimizerD"])
    optimizerD_dur.load_state_dict(checkpoint["optimizerD_dur"])
    schedulerG.load_state_dict(checkpoint["schedulerG"])
    schedulerD.load_state_dict(checkpoint["schedulerD"])
    schedulerD_dur.load_state_dict(checkpoint["schedulerD_dur"])
    now_iteration = checkpoint.get("now_iteration", 0)
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {filepath}")
    return now_iteration, epoch


def load_train_dataset(train_dataset_txtfile_path, hps: HyperParameters):
    train_dataset = TextAudioSpeakerLoader(
        audiopaths_sid_text=train_dataset_txtfile_path, hparams=hps.data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        collate_fn=TextAudioSpeakerCollate(),
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=lambda worker_id: torch.manual_seed(hps.train.seed +
                                                           worker_id),
    )
    print("train dataset size: {}".format(len(train_dataset)))
    return train_loader


def run(load_iteration="", load_chekpoint=False):
    # 学習に必要なパラメータをjsonファイルから読み込む
    hps = HyperParameters.load_from_json("configs/config.json")
    # 乱数のシード設定
    manualSeed = hps.train.seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    hps.model_dir = f"outputs/{hps.model_name}"
    # 出力用ディレクトリがなければ作る
    output_dir = Path(hps.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPUが使用可能かどうか確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセットを読み込む
    train_dataset_txtfile_path = hps.data.training_files
    train_loader = load_train_dataset(train_dataset_txtfile_path, hps)
    print("データセットの読み込みが完了しました。")

    # MSS用ノイズスケールの設定
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    # Duration Discriminatorのインスタンスを生成
    if hps.model.use_duration_discriminator is True:
        logger.info("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.filter_channels,
            3,
            0.1,
            gin_channels=(hps.model.gin_channels
                          if hps.data.n_speakers != 0 else 0),
        ).to(device)
    else:
        net_dur_disc = None

    # Generatorのインスタンスを生成
    net_g = Vits2Generator(
        len(SYMBOLS),
        hps.data.n_mel_channels
        if hps.data.use_mel else hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # hps.model 以下のすべての値を引数に渡す
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.data.use_mel,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
    ).to(device)

    # Discriminatorのインスタンスを生成
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)

    # オプティマイザの設定
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            lr=hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    # 学習率スケジューラ
    def lr_lambda(epoch):
        """
        Learning rate scheduler for warmup and exponential decay.
        - During the warmup period, the learning rate increases linearly.
        - After the warmup period, the learning rate decreases exponentially.
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)
    
    for group in optim_g.param_groups:
        group.setdefault("initial_lr", group.get("lr", hps.train.learning_rate))
    for group in optim_d.param_groups:
        group.setdefault("initial_lr", group.get("lr", hps.train.learning_rate))
    if optim_dur_disc is not None:
        for group in optim_dur_disc.param_groups:
            group.setdefault("initial_lr", group.get("lr", hps.train.learning_rate))

    # 初回作成時は last_epoch を -1 にしておく（resume 時は checkpoint の scheduler state_dict を load する）
    scheduler_last_epoch = -1
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
    else:
        scheduler_dur_disc = None
    logger.info("Start training.")

    # GradScalerはfp16使用時のみ有効化
    scaler = GradScaler(enabled=hps.train.bf16_run)
    print("モデルとオプティマイザの設定が完了しました。")

    if load_checkpoint:
        # チェックポイントからモデルとオプティマイザの状態を読み込む
        checkpoint_path = output_dir.joinpath(load_iteration, "checkpoint.pth")
        if checkpoint_path.exists():
            now_iteration, epoch = load_checkpoint(
                filepath=checkpoint_path,
                netG=net_g,
                netD=net_d,
                netD_dur=net_dur_disc,
                optimizerG=optim_g,
                optimizerD=optim_d,
                optimizerD_dur=optim_dur_disc,
                schedulerG=scheduler_g,
                schedulerD=scheduler_d,
                schedulerD_dur=scheduler_dur_disc,
                device=device,
            )
        else:
            print("Checkpoint not found, starting from scratch.")
            now_iteration = 0
            epoch = 0

    else:
        # チェックポイントがない場合はイテレーションを0から開始
        now_iteration = 0
        epoch = 0

    # lossを記録することで学習過程を追うための変数 学習が安定しるかをグラフから確認できる
    losses_recoded = {
        "adversarial_loss/D": [],
        "adversarial_loss/G": [],
        "adversarial_duration_loss/G": [],
        "adversarial_duration_loss/D": [],
        "mel_reconstruction_loss/G": [],
        "kl_loss/G": [],
        "feature_matching_loss/G": [],
        "dur_loss": []
    }

    kl_start_weight = 0.0
    kl_max_weight = 1.0
    kl_annealing_steps = 100

    def get_kl_weight(step):
        return min(kl_max_weight,
                   kl_start_weight + step / kl_annealing_steps * kl_max_weight)
        
    logger.info("accumulation steps: {}".format(hps.train.accumulation_steps))

    while epoch < hps.train.epochs:
        epoch += 1
        kl_weight = get_kl_weight(epoch)
        now_iteration = train_and_evaluate(
            now_iteration=now_iteration,
            epoch=epoch,
            kl_weight=kl_weight,
            hps=hps,
            nets=(net_g, net_d, net_dur_disc),
            optims=(optim_g, optim_d, optim_dur_disc),
            schedulers=(scheduler_g, scheduler_d,
                        scheduler_dur_disc),
            scaler=scaler,
            loaders=train_loader,
            losses_recorded=losses_recoded,
            device=device,
        )
        if now_iteration >= hps.train.iterations + 1:
            print("学習終了: now_iterationがtotal_iterationsを超えました。")
            break


def train_and_evaluate(
    now_iteration,
    epoch,
    kl_weight,
    hps: HyperParameters,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    losses_recorded,
    device,
):
    # ネットワークの取得
    net_g, net_d, net_dur_disc = nets
    net_g: Vits2Generator
    net_d: MultiPeriodDiscriminator
    net_dur_disc: DurationDiscriminator
    # オプティマイザの取得
    optim_g, optim_d, optim_dur_disc = optims

    # スケジューラの取得
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    # データローダの取得
    train_loader = loaders

    # ネットワークを学習モードに
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    # 進捗バーの設定
    loader = tqdm(train_loader, desc=f"Epoch {epoch}")
    # オプティマイザの勾配を初期化
    optim_g.zero_grad(set_to_none=True)
    optim_d.zero_grad(set_to_none=True)
    if optim_dur_disc is not None:
        optim_dur_disc.zero_grad(set_to_none=True)
    
    for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
    ) in enumerate(loader):
        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = (net_g.mas_noise_scale_initial -
                                       net_g.noise_scale_delta * epoch)
            net_g.mas_noise_scale_initial = max(current_mas_noise_scale, 0.0)
        # データをデバイスに転送
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        speakers = speakers.to(device)

        with autocast(device_type=str(device),
                      enabled=hps.train.bf16_run,
                      dtype=torch.bfloat16):
            (
                y_hat,
                l_length,
                _,  # attn
                ids_slice,
                x_mask,
                z_mask,
                (_, z_p, m_p, logs_p, _, logs_q),  # (z, z_p, m_p, logs_p, m_q, logs_q)
                (hidden_x, logw, logw_),  # , logw_sdp),
                _,  # g
            ) = net_g(x, x_lengths, spec, spec_lengths, speakers)
            # 入力がmelではない場合にmel計算
            if not hps.data.use_mel:
                mel = spec_to_mel(
                    spec,
                    n_fft=hps.data.filter_length,
                    n_mels=hps.data.n_mel_channels,
                    sample_rate=hps.data.sampling_rate,
                    f_min=hps.data.mel_fmin,
                    f_max=hps.data.mel_fmax,
                )
            else:
                mel = spec

            y_hat_mel = wav_to_mel(
                y_hat.squeeze(1).float(),
                n_fft=hps.data.filter_length,
                num_mels=hps.data.n_mel_channels,
                sampling_rate=hps.data.sampling_rate,
                hop_size=hps.data.hop_length,
                win_size=hps.data.win_length,
                fmin=hps.data.mel_fmin,
                fmax=hps.data.mel_fmax,
            )

            # slice
            y_mel = slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            # real wav
            y = slice_segments(y, ids_slice * hps.data.hop_length,
                               hps.train.segment_size)

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(device_type=str(device),
                          enabled=hps.train.bf16_run,
                          dtype=torch.bfloat16):
                loss_disc, _, _ = discriminator_loss(  # loss_disc, losses_disc_r, losses_disc_g
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
                loss_disc_all = loss_disc_all / hps.train.accumulation_steps

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                )
                with autocast(device_type=str(device),
                              enabled=hps.train.bf16_run,
                              dtype=torch.bfloat16):
                    (
                        loss_dur_disc,
                        _,  # losses_dur_disc_r
                        _,  # losses_dur_disc_g
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
                    loss_dur_disc_all = loss_dur_disc_all / hps.train.accumulation_steps
                    # Duration Discriminatorの更新
                    if (batch_idx + 1) % hps.train.accumulation_steps == 0 or len(train_loader) == (batch_idx + 1):
                        scaler.scale(loss_dur_disc_all).backward()
                        scaler.unscale_(optim_dur_disc)
                        clip_grad_value_(net_dur_disc.parameters(), None)
                        scaler.step(optim_dur_disc)
                        scaler.update()
                        optim_dur_disc.zero_grad(set_to_none=True)

                        # -- メモリ解放処理 ---
                        try:
                            del loss_dur_disc, y_dur_hat_r, y_dur_hat_g
                        except Exception:
                            pass
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

        # Discriminatorの更新
        scaler.scale(loss_disc_all).backward()
        if (batch_idx + 1) % hps.train.accumulation_steps == 0 or len(train_loader) == (batch_idx + 1):
            scaler.unscale_(optim_d)
            if getattr(hps.train, "bf16_run", False):
                torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(),
                                            max_norm=200)
            clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)
            scaler.update()
            optim_d.zero_grad(set_to_none=True)
            
            # --- メモリ解放処理 ---
            try:
               del loss_disc, y_d_hat_r, y_d_hat_g
            except Exception:
               pass
            for _v in ("y_hat", "y_d_hat_r", "y_d_hat_g"):
               if _v in locals():
                   try:
                       del locals()[_v]
                   except Exception:
                       pass
            gc.collect()
            torch.cuda.empty_cache()

        # Generator
        with autocast(device_type=str(device),
                      enabled=hps.train.bf16_run,
                      dtype=torch.bfloat16):
            _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(
                device_type=str(device),
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16
            ):
                loss_dur = torch.sum(l_length.float())
                # 再構成誤差
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # generator loss
                loss_gen, _ = generator_loss(y_d_hat_g)  # loss_gen, losses_gen
                # kl loss
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                # feature loss
                loss_fm = feature_loss(fmap_r, fmap_g)
                # total loss
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl * kl_weight
                # sdp generator loss
                if net_dur_disc is not None:
                    loss_dur_gen, _ = generator_loss(y_dur_hat_g)  # loss_dur_gen, losses_dur_gen
                    loss_gen_all += loss_dur_gen
                
                loss_gen_all = loss_gen_all / hps.train.accumulation_steps
        
        # Generatorの更新
        scaler.scale(loss_gen_all).backward()
        if (batch_idx + 1) % hps.train.accumulation_steps == 0 or len(train_loader) == (batch_idx + 1):
            scaler.unscale_(optim_g)
            if getattr(hps.train, "bf16_run", False):
                torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
            clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()
            optim_g.zero_grad(set_to_none=True)
            y_mel = y_mel.detach().cpu().numpy().astype(np.float32)
            y_hat_mel = y_hat_mel.detach().cpu().numpy().astype(np.float32)
            # テキストの正確さを計算
            text_correct = torch.eq(x_mask, hidden_x).sum().item()
            text_ratio = text_correct / torch.numel(x_mask)
            
            # メモリ解放
            try:
                # 大きなテンソルを削除
                del y_hat
            except Exception:
                pass
            # 一時変数の削除
            for _v in ("l_length", "ids_slice", "x_mask", "z_mask", "fmap_r", "fmap_g", "z_p", "m_p", "logs_p", "m_q",
            "logs_q", "hidden_x", "logw", "logw_"):
                if _v in locals():
                    try:
                        del locals()[_v]
                    except Exception:
                        pass
            
            # ガベージコレクションを実行
            gc.collect()
            torch.cuda.empty_cache()

        if (batch_idx + 1) % hps.train.accumulation_steps == 0 or len(train_loader) == (batch_idx + 1):
            # ####stdoutへlossを出力する#####
            loss_stdout = {
                "adversarial_loss/D": loss_disc_all.item(),
                "adversarial_loss/G": loss_gen.item(),
                "adversarial_duration_loss/D": loss_dur_disc_all.item(),
                "adversarial_duration_loss/G": loss_dur_gen.item(),
                "mel_reconstruction_loss/G": loss_mel.item(),
                "kl_loss/G": min(loss_kl.item(), 200),
                "feature_matching_loss/G": loss_fm.item(),
                "dur_loss": loss_dur.item(),
            }

            if now_iteration % 10 == 0:
                for key, value in loss_stdout.items():
                    print(f" {key}:{value:.5f}", end="")
                print(f" text_accuracy:{text_ratio:.5f}", end="")
                # print(f" real_loss/D:{torch.mean(torch.tensor(real_losses)):.5f}", end="")
                # print(f" fake_loss/D:{torch.mean(torch.tensor(fake_losses)):.5f}", end="")
                # print(f" real_mean/D:{torch.mean(torch.tensor(real_means)):.3f}", end="")
                # print(f" fake_mean/D:{torch.mean(torch.tensor(fake_means)):.3f}", end="")
                print("")

                # メルスペクトログラムの描画
                plot_melspectrogram_cv2(y_mel, title="Real")
                plot_melspectrogram_cv2(y_hat_mel, title="Fake")

            # lossを記録
            for key, value in loss_stdout.items():
                losses_recorded[key].append(value)

            # ####学習状況をファイルに出力#####
            if now_iteration % hps.train.log_interval == 0 and now_iteration != 0:
                out_dir = os.path.join(hps.model_dir, f"iteration{now_iteration}")
                # 出力用ディレクトリがなければ作る
                os.makedirs(out_dir, exist_ok=True)

                # ####チェックポイントを保存#####
                save_checkpoint(
                    netG=net_g,
                    netD=net_d,
                    netD_dur=net_dur_disc,
                    optimizerG=optim_g,
                    optimizerD=optim_d,
                    optimizerD_dur=optim_dur_disc,
                    schedulerG=scheduler_g,
                    schedulerD=scheduler_d,
                    schedulerD_dur=scheduler_dur_disc,
                    now_iteration=now_iteration,
                    epoch=epoch,
                    filepath=os.path.join(out_dir, "checkpoint.pth"),
                )

                # ####学習済みモデル（CPU向け）を出力#####
                # generatorを出力
                net_g.eval()
                torch.save(
                    net_g.to("cpu").state_dict(),
                    os.path.join(out_dir, "netG_cpu.pth"))
                net_g.to(device)
                net_g.train()
                # discriminatorを出力
                net_d.eval()
                torch.save(
                    net_d.to("cpu").state_dict(),
                    os.path.join(out_dir, "netD_cpu.pth"))
                net_d.to(device)
                net_d.train()
                # duration discriminatorを出力
                net_dur_disc.eval()
                torch.save(
                    net_dur_disc.to("cpu").state_dict(),
                    os.path.join(out_dir, "netD_dur_cpu.pth"),
                )
                net_dur_disc.to(device)
                net_dur_disc.train()

                # ####lossのグラフを出力#####
                plt.clf()
                plt.figure(figsize=(16, 6))
                plt.subplots_adjust(wspace=0.4, hspace=0.6)
                for i, (loss_name, loss_list) in enumerate(losses_recorded.items(),
                                                        0):
                    plt.subplot(2, 4, i + 1)
                    plt.title(loss_name)
                    plt.plot(loss_list, label="loss")
                    plt.xlabel("iterations")
                    plt.ylabel("loss")
                    plt.legend()
                    plt.grid()
                plt.savefig(os.path.join(out_dir, "loss.png"))
                plt.close()

            now_iteration += 1
            torch.cuda.empty_cache()

        # イテレーション数が上限に達したらループを抜ける
        if now_iteration >= hps.train.iterations + 1:
            break

    # 学習率を更新
    scheduler_d.step()
    scheduler_dur_disc.step()
    scheduler_g.step()

    return now_iteration


def plot_melspectrogram_cv2(mel_spec, title="Mel Spectrogram"):
    # mel_spec: torch.Tensor [batch, n_mels, time] or [n_mels, time]
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.detach().cpu().numpy().astype(np.float32)
    # バッチ次元があれば最初のサンプルのみ
    if mel_spec.ndim == 3:
        mel_spec = mel_spec[0]
    # 正規化（0-255にスケール）
    mel_min, mel_max = mel_spec.min(), mel_spec.max()
    mel_img = (mel_spec - mel_min) / (mel_max - mel_min + 1e-8)
    # 画像サイズを3倍に拡大
    mel_img = cv2.resize(
        mel_img,
        (mel_img.shape[1] * 3, mel_img.shape[0] * 3),
        interpolation=cv2.INTER_LINEAR,
    )
    # 画像を表示
    cv2.imshow(title, mel_img)
    cv2.waitKey(1)  # 1ミリ秒待機してウィンドウを更新


if __name__ == "__main__":
    main()
