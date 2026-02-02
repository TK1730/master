# 標準ライブラリ
import gc
import os
import platform
import random
import sys
from pathlib import Path

# サードパーティライブラリ
import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ローカルモジュール
from data_utils import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from module import utils
from module.vits2_discriminator import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
)
from module.vits2_generator import Vits2Generator
from nlp.symbols import SYMBOLS
from utils.hyper_parameters import HyperParameters
from utils.logger import logger
from utils.mel_processing import wav_to_mel
from utils.model import clip_grad_value_, slice_segments

global_step = 0


def main():
    run(load_config_path="configs/config.json")


def load_train_dataset(train_dataset_txtfile_path: str, hps: HyperParameters):
    """
    訓練データセットを読み込み、DataLoaderを作成する

    この関数は訓練用のテキスト・音声・話者情報を含むデータセットを読み込み、
    PyTorchのDataLoaderとして返します。DataLoaderは設定されたバッチサイズ、
    シャッフル、および再現性のための乱数シードを使用して初期化されます。

    Args:
        train_dataset_txtfile_path (str): 訓練データセットのファイルパス
            各行に「音声ファイルパス|話者ID|テキスト」の形式で記述されたテキストファイル
        hps (HyperParameters): ハイパーパラメータオブジェクト
            データセット設定(hps.data)と訓練設定(hps.train)を含む

    Returns:
        DataLoader: 訓練用データローダー
            バッチごとにテキスト、音声、話者情報を提供
    """
    # テキスト・音声・話者情報を含むデータセットを作成
    train_dataset = TextAudioSpeakerLoader(
        audiopaths_sid_text=train_dataset_txtfile_path, hparams=hps.data)

    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,  # バッチサイズ
        collate_fn=TextAudioSpeakerCollate(),  # バッチ内のデータを整形する関数
        num_workers=0,  # データロード用のサブプロセス数（0=メインプロセスのみ）
        shuffle=True,  # エポックごとにデータをシャッフル
        pin_memory=True,  # GPU転送を高速化（CUDAメモリに固定）
        worker_init_fn=lambda worker_id: torch.manual_seed(
            hps.train.seed + worker_id),  # 各ワーカーの乱数シードを設定して再現性を確保
    )
    print("train dataset size: {}".format(len(train_dataset)))
    return train_loader


def load_test_dataset(test_dataset_txtfile_path: str, hps: HyperParameters):
    """
    テストデータセットを読み込む

    Args:
        test_dataset_txtfile_path (str): テストデータセットのファイルパス
        hps (HyperParameters): ハイパーパラメータ

    Returns:
        DataLoader: テストデータローダー
    """
    # テキスト・音声・話者情報を含むデータセットを作成
    test_dataset = TextAudioSpeakerLoader(
        audiopaths_sid_text=test_dataset_txtfile_path, hparams=hps.data)

    # DataLoaderを作成
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # テストは1サンプルずつ
        collate_fn=TextAudioSpeakerCollate(),
        num_workers=0,
        shuffle=False,
        pin_memory=True,
    )
    print("test dataset size: {}".format(len(test_dataset)))
    return test_loader


def cleanup_memory():
    """
    メモリを解放する
    GPUメモリとCPUメモリの両方をクリーンアップ
    """
    gc.collect()
    torch.cuda.empty_cache()


def create_loss_dict(
    loss_disc_all,
    loss_gen,
    loss_dur_disc_all,
    loss_dur_gen,
    loss_mel,
    loss_kl,
    loss_fm,
    loss_dur,
    net_dur_disc
):
    """
    損失値を辞書形式で作成

    Args:
        各種損失値とnet_dur_disc

    Returns:
        dict: 損失値の辞書
    """
    return {
        "adversarial_loss/D": loss_disc_all.item(),
        "adversarial_loss/G": loss_gen.item(),
        "adversarial_duration_loss/D": (
            loss_dur_disc_all.item() if net_dur_disc is not None else 0.0
        ),
        "adversarial_duration_loss/G": (
            loss_dur_gen.item() if net_dur_disc is not None else 0.0
        ),
        "mel_reconstruction_loss/G": loss_mel.item(),
        "kl_loss/G": min(loss_kl.item(), 200),
        "feature_matching_loss/G": loss_fm.item(),
        "dur_loss": loss_dur.item(),
    }


def lr_lambda(epoch: int, warmup_epochs: int, lr_decay: float) -> float:
    """
    学習率スケジューラのための係数を計算

    ウォームアップ期間中は学習率を線形に増加させ、
    その後は指数関数的に減衰させます。

    Args:
        epoch (int): 現在のエポック数
        warmup_epochs (int): ウォームアップエポック数
        lr_decay (float): 学習率減衰率

    Returns:
        float: 学習率の係数（初期学習率に対する倍率）
    """
    if epoch < warmup_epochs:
        # ウォームアップ期間: 線形に増加
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        # ウォームアップ後: 指数関数的に減衰
        return lr_decay ** (epoch - warmup_epochs)


def run(load_config_path=str):
    # 学習に必要なパラメータをjsonファイルから読み込む
    hps = HyperParameters.load_from_json(load_config_path)
    # 乱数のシード設定
    random.seed(hps.train.seed)
    torch.manual_seed(hps.train.seed)

    # iteration数の初期化
    global global_step
    epoch_str = 0

    # モデル保存用ディレクトリの設定
    hps.model_dir = f"outputs/{hps.model_name}"

    # 出力用ディレクトリがなければ作る
    output_dir = Path(hps.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPUが使用可能かどうか確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # トレインデータセットを読み込む
    train_dataset_txtfile_path = hps.data.training_files
    train_loader = load_train_dataset(train_dataset_txtfile_path, hps)

    # テストデータセットを読み込む
    test_dataset_txtfile_path = hps.data.validation_files
    test_loader = load_test_dataset(test_dataset_txtfile_path, hps)
    print("データセットの読み込みが完了しました。")

    # trainのwriter
    train_writer = SummaryWriter(log_dir=f"{hps.model_dir}/logs/train")
    test_writer = None

    # MAS用ノイズスケールの設定
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 1.0
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

    if hps.model.use_spk_conditoned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder "
                "to train multi-speaker model"
            )
    else:
        logger.info("Using normal encoder for VITS1")

    # Generatorのインスタンスを生成
    net_g = Vits2Generator(
        n_vocab=len(SYMBOLS),
        spec_channels=hps.data.spec_channels,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        inter_channels=hps.model.inter_channels,
        n_speakers=hps.data.n_speakers,
        # hps.model 以下のすべての値を引数に渡す
        gin_channels=hps.model.gin_channels,
        # TextEncoder用パラメータ
        text_enc_hidden_channels=hps.model.text_enc_hidden_channels,
        text_enc_filter_channels=hps.model.text_enc_filter_channels,
        text_enc_n_heads=hps.model.text_enc_n_heads,
        text_enc_n_layers=hps.model.text_enc_n_layers,
        text_enc_kernel_size=hps.model.text_enc_kernel_size,
        text_enc_p_dropout=hps.model.text_enc_p_dropout,
        # PosteriorEncoder用パラメータ
        post_enc_n_layers=hps.model.post_enc_n_layers,
        post_enc_kernel_size=hps.model.post_enc_kernel_size,
        post_enc_dilation_rate=hps.model.post_enc_dilation_rate,
        # Decoder (Generator)用パラメータ
        decoder_resblock_type=hps.model.decoder_resblock_type,
        decoder_resblock_kernel_sizes=hps.model.decoder_resblock_kernel_sizes,
        decoder_resblock_dilation_sizes=(
            hps.model.decoder_resblock_dilation_sizes
        ),
        decoder_upsample_rates=hps.model.decoder_upsample_rates,
        decoder_upsample_initial_channel=(
            hps.model.decoder_upsample_initial_channel
        ),
        decoder_upsample_kernel_sizes=hps.model.decoder_upsample_kernel_sizes,
        # Flow用パラメータ
        flow_n_layers=hps.model.flow_n_layers,
        flow_transformer_n_layers=hps.model.flow_transformer_n_layers,
        flow_share_parameter=hps.model.flow_share_parameter,
        flow_kernel_size=hps.model.flow_kernel_size,
        use_transformer_flow=hps.model.use_transformer_flow,
        # DurationPredictor用パラメータ
        sdp_filter_channels=hps.model.sdp_filter_channels,
        sdp_kernel_size=hps.model.sdp_kernel_size,
        sdp_p_dropout=hps.model.sdp_p_dropout,
        sdp_n_flows=hps.model.sdp_n_flows,
        dp_filter_channels=hps.model.dp_filter_channels,
        dp_kernel_size=hps.model.dp_kernel_size,
        dp_p_dropout=hps.model.dp_p_dropout,
        # その他の設定:
        use_sdp=hps.model.use_sdp,
        use_spk_conditoned_encoder=hps.model.use_spk_conditoned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
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

    if utils.is_resuming(hps.model_dir):
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(
                    hps.model_dir, "DUR_*.pth"
                ),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer,
            )
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
            utils.checkpoints.get_latest_checkpoint_path(
                hps.model_dir, "G_*.pth"
            ),
            net_g,
            optim_g,
            skip_optimizer=hps.train.skip_optimizer,
        )
        _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
            utils.checkpoints.get_latest_checkpoint_path(
                hps.model_dir, "D_*.pth"
            ),
            net_d,
            optim_d,
            skip_optimizer=hps.train.skip_optimizer,
        )
        if not optim_g.param_groups[0].get("initial_lr"):
            optim_g.param_groups[0]["initial_lr"] = g_resume_lr
        if not optim_d.param_groups[0].get("initial_lr"):
            optim_d.param_groups[0]["initial_lr"] = d_resume_lr

        epoch_str = max(epoch_str, 1)
        global_step = int(
            utils.get_steps(
                utils.checkpoints.get_latest_checkpoint_path(
                    hps.model_dir, "G_*.pth"
                )
            )
        )
        logger.info(
            f"******************Found the model. Current epoch is {epoch_str},"
            f" global step is {global_step}*********************"
        )
    else:
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(hps.model_dir, "G_0.safetensors"), net_g
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(hps.model_dir, "D_0.safetensors"), net_d
            )
            if net_dur_disc is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(
                        hps.model_dir,
                        "DUR_0.safetensors"
                    ),
                    net_dur_disc
                )
            logger.info("Loaded the pretrained models.")
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, "
                "so we will train from scratch."
            )
        finally:
            epoch_str = 1
            global_step = 0

    # 学習率スケジューラの設定
    # オプティマイザのパラメータグループに初期学習率を設定
    for group in optim_g.param_groups:
        group.setdefault(
            "initial_lr",
            group.get("lr", hps.train.learning_rate)
        )
    for group in optim_d.param_groups:
        group.setdefault(
            "initial_lr",
            group.get("lr", hps.train.learning_rate)
        )
    if optim_dur_disc is not None:
        for group in optim_dur_disc.param_groups:
            group.setdefault(
                "initial_lr",
                group.get(
                    "lr",
                    hps.train.learning_rate
                )
            )

    # 初回作成時は last_epoch を -1 にしておく
    # (resume 時は checkpoint の scheduler state_dict を load する)
    scheduler_last_epoch = -1
    # 学習率スケジューラを作成（ウォームアップ＋減衰）
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g,
        lr_lambda=lambda epoch: lr_lambda(
            epoch, hps.train.warmup_epochs, hps.train.lr_decay
        ),
        last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d,
        lr_lambda=lambda epoch: lr_lambda(
            epoch, hps.train.warmup_epochs, hps.train.lr_decay
        ),
        last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc,
            lr_lambda=lambda epoch: lr_lambda(
                epoch, hps.train.warmup_epochs, hps.train.lr_decay
            ),
            last_epoch=scheduler_last_epoch
        )
    else:
        scheduler_dur_disc = None

    # GradScalerはfp16使用時のみ有効化
    scaler = GradScaler(enabled=hps.train.bf16_run)
    print("モデルとオプティマイザの設定が完了しました。")

    diff = abs(
        epoch_str * len(train_loader)
        - (hps.train.epochs + 1) * len(train_loader)
    )

    # tqdmの出力先を標準出力に変更
    if platform.system() == "Windows":
        SAFE_STDOUT = open(
            sys.stdout.fileno(),
            'w',
            encoding='utf-8',
            closefd=False
        )
    else:
        SAFE_STDOUT = sys.stdout

    pbar = tqdm(
        total=global_step + diff,
        initial=global_step,
        smoothing=0.05,
        file=SAFE_STDOUT,
        dynamic_ncols=True
    )

    initial_step = global_step
    logger.info("Start training.")

    logger.info(
        "accumulation steps: {}".format(hps.train.accumulation_steps)
    )

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            epoch=epoch,
            hps=hps,
            nets=(net_g, net_d, net_dur_disc),
            optims=(optim_g, optim_d, optim_dur_disc),
            schedulers=(
                scheduler_g,
                scheduler_d,
                scheduler_dur_disc
            ),
            scaler=scaler,
            loaders=(train_loader, test_loader),
            writers=(train_writer, test_writer),
            pbar=pbar,
            initial_step=initial_step,
            device=device,
        )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()

        if epoch == hps.train.epochs:
            # 最後のエポックで学習を終了する
            assert optim_g is not None
            utils.checkpoints.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            assert optim_d is not None
            utils.checkpoints.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                utils.checkpoints.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        hps.model_dir, "DurDisc_{}.pth".format(global_step)
                    ),
                )
            utils.safetensors.save_safetensors(
                net_g,
                epoch,
                os.path.join(
                    hps.model_dir,
                    f"G_{hps.model_name}_e{epoch}_s{global_step}.safetensors",
                ),
                for_infer=True
            )

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    epoch,
    hps: HyperParameters,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    writers,
    pbar: tqdm,
    initial_step,
    device,
):
    # ネットワークの取得
    net_g, net_d, net_dur_disc = nets
    net_g: Vits2Generator
    net_d: MultiPeriodDiscriminator
    if net_dur_disc is not None:
        net_dur_disc: DurationDiscriminator

    # オプティマイザの取得
    optim_g, optim_d, optim_dur_disc = optims
    # スケジューラの取得
    scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
    # データローダの取得
    train_loader, test_loader = loaders
    if writers is not None:
        writer_train, _ = writers
    global global_step

    # ネットワークを学習モードに
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

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
    ) in enumerate(train_loader):
        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = (net_g.mas_noise_scale_initial -
                                       net_g.noise_scale_delta * epoch)
            net_g.mas_noise_scale_initial = max(current_mas_noise_scale, 0.0)
        # データをデバイスに転送
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        speakers = speakers.to(device)

        with autocast(
            device_type=str(device),
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16
        ):
            (
                y_hat,
                l_length,
                _,  # attn
                ids_slice,
                x_mask,
                z_mask,
                (_, z_p, m_p, logs_p, _, logs_q),
                (hidden_x, logw, logw_),
                g,
            ) = net_g(x, x_lengths, spec, spec_lengths, speakers)
            # real
            y_mel = wav_to_mel(
                y.squeeze(1).float(),
                n_fft=hps.data.filter_length,
                num_mels=hps.data.n_mel_channels,
                sampling_rate=hps.data.sampling_rate,
                hop_size=hps.data.hop_length,
                win_size=hps.data.win_length,
                fmin=hps.data.mel_fmin,
                fmax=hps.data.mel_fmax,
            )

            # fake
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
            y = slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            y_mel = slice_segments(
                y_mel,
                ids_slice,
                hps.train.segment_size // hps.data.hop_length,
            )

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(device_type=str(device),
                          enabled=hps.train.bf16_run,
                          dtype=torch.bfloat16):
                # loss_disc, losses_disc_r, losses_disc_g
                loss_disc, _, _ = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g
                )
                loss_disc_all = loss_disc
                loss_disc_all = loss_disc_all / hps.train.accumulation_steps

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw.detach(),
                    logw_.detach(),
                    g.detach() if g is not None else None,
                )
                with autocast(device_type=str(device),
                              enabled=hps.train.bf16_run,
                              dtype=torch.bfloat16):
                    # loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g
                    loss_dur_disc, _, _ = discriminator_loss(
                        y_dur_hat_r,
                        y_dur_hat_g,
                    )
                    loss_dur_disc_all = loss_dur_disc
                    loss_dur_disc_all = (
                        loss_dur_disc_all / hps.train.accumulation_steps
                    )
                # Duration Discriminatorの更新 accumと最後のbatchで更新
                if (
                    (batch_idx + 1) % hps.train.accumulation_steps == 0
                    or len(train_loader) == (batch_idx + 1)
                ):
                    optim_dur_disc.zero_grad(set_to_none=True)
                    scaler.scale(loss_dur_disc_all).backward()
                    scaler.unscale_(optim_dur_disc)
                    clip_grad_value_(net_dur_disc.parameters(), None)
                    scaler.step(optim_dur_disc)
                    scaler.update()

                    # -- メモリ解放処理 ---
                    try:
                        del y_dur_hat_r, y_dur_hat_g
                    except Exception:
                        pass

        # Discriminatorの更新 accumと最後のbatchで更新
        if (
            (batch_idx + 1) % hps.train.accumulation_steps == 0
            or len(train_loader) == (batch_idx + 1)
        ):
            optim_d.zero_grad(set_to_none=True)
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            if getattr(hps.train, "bf16_run", False):
                torch.nn.utils.clip_grad_norm_(
                    parameters=net_d.parameters(),
                    max_norm=200
                )
            grad_norm_d = clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            # --- メモリ解放処理 ---
            try:
                del y_d_hat_r, y_d_hat_g
            except Exception:
                pass

        # Generator
        with autocast(
            device_type=str(device),
            enabled=hps.train.bf16_run,
            dtype=torch.bfloat16
        ):
            # Generator
            _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(
                    hidden_x,
                    x_mask,
                    logw_,
                    logw,
                    g if g is not None else None
                )
            with autocast(
                device_type=str(device),
                enabled=hps.train.bf16_run,
                dtype=torch.bfloat16
            ):
                loss_dur = torch.sum(l_length.float())
                # 再構成誤差
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # generator loss
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                # kl loss
                loss_kl = kl_loss(
                    z_p, logs_q, m_p, logs_p, z_mask
                ) * hps.train.c_kl
                # feature loss
                loss_fm = feature_loss(fmap_r, fmap_g)
                # total loss
                loss_gen_all = (
                    loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                )
                # sdp generator loss
                if net_dur_disc is not None:
                    # loss_dur_gen, losses_dur_gen
                    loss_dur_gen, _ = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

                loss_gen_all = loss_gen_all / hps.train.accumulation_steps

        # Generatorの更新
        if (
            (batch_idx + 1) % hps.train.accumulation_steps == 0
            or len(train_loader) == (batch_idx + 1)
        ):
            optim_g.zero_grad(set_to_none=True)
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            if getattr(hps.train, "bf16_run", False):
                torch.nn.utils.clip_grad_norm_(
                    parameters=net_g.parameters(), max_norm=500
                )
            grad_norm_g = clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            y_mel = y_mel.detach().cpu().numpy().astype(np.float32)
            y_hat_mel = y_hat_mel.detach().cpu().numpy().astype(np.float32)
            # テキストの正確さを計算
            # text_correct = torch.eq(x_mask, hidden_x).sum().item()
            # text_ratio = text_correct / torch.numel(x_mask)

            # メモリ解放
            try:
                # 大きなテンソルを削除
                del y_hat
            except Exception:
                pass
            cleanup_memory()

        if global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            loss_msg = (
                f"Step: {global_step}, LR: {lr:.6f}, "
                f"Disc: {loss_disc.item():.4f}, Gen: {loss_gen.item():.4f}, "
                f"FM: {loss_fm.item():.4f}, Mel: {loss_mel.item():.4f}, "
                f"Dur: {loss_dur.item():.4f}, KL: {loss_kl.item():.4f}"
            )
            # Duration Discriminatorのロスがある場合は追加
            if net_dur_disc is not None:
                loss_msg += (
                    f", DurDisc: {loss_dur_disc.item():.4f}, "
                    f"DurGen: {loss_dur_gen.item():.4f}"
                )
            logger.info(loss_msg)

            scaler_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scaler_dict.update({
                "loss/g/mel": loss_mel,
                "loss/g/dur": loss_dur,
                "loss/g/kl": loss_kl,
                "loss/g/fm": loss_fm,
            })
            utils.summarize(
                writer=writer_train,
                global_step=global_step,
                scalars=scaler_dict,
            )

            # 画像描画
            plot_melspectrogram_cv2(y_mel, "y_mel")
            plot_melspectrogram_cv2(y_hat_mel, "y_hat_mel")

        if (
            global_step % hps.train.eval_interval == 0
            and global_step != 0
            and initial_step != global_step
        ):
            evaluate(
                hps=hps,
                generator=net_g,
                test_loader=test_loader,
                device=device,
                output_dir=hps.model_dir,
            )
            utils.checkpoints.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth"),
            )
            utils.checkpoints.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth"),
            )
            if net_dur_disc is not None:
                utils.checkpoints.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, f"DurDisc_{global_step}.pth"),
                )
            keep_ckpts = 1  # 保存するチェックポイント数
            if keep_ckpts > 0:
                utils.checkpoints.clean_checkpoints(
                    hps.model_dir,
                    keep_ckpts,
                    sort_by_time=True,
                )
            utils.safetensors.save_safetensors(
                net_g,
                epoch
                os.path.join(
                    hps.model_dir,
                    f"{hps.model_name}_e{epoch}_s{global_step}_G.safetensors",
                ),
                for_infer=True,
            )
        if (
            (batch_idx + 1) % hps.train.accumulation_steps == 0
            or len(train_loader) == (batch_idx + 1)
        ):
            global_step += 1
            if pbar is not None:
                pbar.set_description(
                    f"Epoch {epoch}"
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)"
                    f"/{hps.train.epochs}"
                )
                pbar.update(1)
    # メモリの削除
    cleanup_memory()
    if pbar is None:
        logger.info(f"===> Epoch: {epoch}, step: {global_step}")


def evaluate(
    hps: HyperParameters,
    generator: Vits2Generator,
    test_loader: DataLoader,
    device: str,
    output_dir: str,
):
    """
    モデルの評価(L1 loss計算)とテスト音声の合成を同時に行う

    Args:
        hps: ハイパーパラメータ
        generator: Generatorモデル
        test_loader: テストデータローダー
        device: 使用デバイス
        output_dir: 出力ディレクトリ

    Returns:
        float: 平均L1 loss
    """
    generator.eval()
    count = 0
    print()
    logger.info("Evaluating...")

    # 音声保存用ディレクトリ
    audio_out_dir = os.path.join(output_dir, "test_audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
        ) in enumerate(test_loader):
            # デバイス転送
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            speakers = speakers.to(device)

            # 音声合成
            y_hat, *_ = generator.infer(
                x,
                x_lengths,
                sid=speakers,
                max_len=1000,
            )

            # 音声の保存（合成用）
            if idx < 2:
                audio = y_hat.squeeze().cpu().numpy()
                output_path = os.path.join(audio_out_dir, f"sample_{idx}.wav")
                sf.write(
                    output_path,
                    audio,
                    hps.data.sampling_rate,
                    subtype='PCM_16'
                )
            count += 1

    generator.train()


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
        (mel_img.shape[1] * 4, mel_img.shape[0] * 4),
        interpolation=cv2.INTER_LINEAR,
    )
    # 画像を表示
    cv2.imshow(title, mel_img)
    cv2.waitKey(1)  # 1ミリ秒待機してウィンドウを更新


if __name__ == "__main__":
    main()
