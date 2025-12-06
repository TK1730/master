import librosa
from pathlib import Path
import numpy as np
import torch
import pyworld as pw
import librosa
import soundfile as sf

from utils import config
from utils import functions
from model.lstm_net_rev import LSTM_net


def process_and_save(input_path, output_path, net_ppg, net_f0mat, net_mcp,
                     net_cap, device):
    """
    擬似ささやき声から有声発話への変換を行い、指定されたパスに保存する。

    Parameters:
        input_path (Path): 入力音声ファイルのパス
        output_path (Path): 出力音声ファイルの保存先パス
        net_ppg, net_f0mat, net_mcp, net_cap: 学習済みモデル
        device: 使用するデバイス (GPU/CPU)
    """
    with torch.inference_mode():
        wav, sr = librosa.load(str(input_path), sr=config.sr)
        wav = functions.loudness_normalize(wav, config.lufs_mix, load=False)
        D = librosa.stft(
            y=wav, n_fft=config.n_fft, hop_length=config.hop_length,
            win_length=config.win_length, pad_mode='reflect'
        ).T
        sp, phase = librosa.magphase(D)
        msp = np.matmul(sp, functions.mel_filter)
        lmsp = functions.dynamic_range_compression(msp)
        lmsp = torch.from_numpy(lmsp).unsqueeze(0).to(
            torch.float32).to(device)

        # PPG
        ppg = net_ppg(lmsp)

        # PPGを結合
        with torch.no_grad():
            asr_output = net_ppg(lmsp).detach().to("cpu")
            asr_probs = torch.softmax(asr_output, dim=-1)
            ppg = torch.eye(36)[torch.argmax(asr_probs, dim=2)].to(device)
        lmsp = torch.cat([lmsp, ppg], dim=2)

        # 各特徴量を推定
        rcap = net_cap(lmsp).detach().to('cpu').numpy()[0]
        rf0 = net_f0mat(lmsp).detach().to('cpu').numpy()[0]
        rmcp = net_mcp(lmsp).detach().to('cpu').numpy()[0]

        cap = np.log(rcap.clip(1e-5, None))
        ap = pw.decode_aperiodicity(cap.astype(np.float64), fs=config.sr,
                                    fft_size=config.n_fft)
        cp = pw.decode_spectral_envelope(rmcp.astype(np.float64),
                                         fs=config.sr,
                                         fft_size=config.n_fft)

        # f0mat melfilter
        f0matave = (rf0.clip(0, None).T) / (rf0.clip(0, None)).sum(axis=1).T
        rmel = (f0matave.T * librosa.hz_to_mel(
            functions.mel_freqs, htk=False)).T.sum(axis=0)
        rmel = np.convolve(rmel, np.ones(5) / 5, mode='same')
        f0 = librosa.mel_to_hz(rmel, htk=False) / 10

        out_wav = pw.synthesize(
            f0.astype(np.float64).reshape(-1),
            cp.astype(np.float64),
            ap.astype(np.float64),
            fs=config.sr,
            frame_period=config.hop_length * 1000 / config.sr
        )
        out_wav = functions.loudness_normalize(
            out_wav, config.lufs_jvs, load=False)

        # 出力先フォルダを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 音声ファイルを保存
        sf.write(str(output_path), out_wav, config.sr)
        print(f"Processed and saved: {output_path}")


def load_models(device):
    """
    学習済みモデルをロードして返す。

    Parameters:
        device: 使用するデバイス (GPU/CPU)

    Returns:
        dict: ロードされたモデルの辞書
    """
    models = {}

    # モデルフォルダのパス
    model_dir = Path('model')

    # PPGモデル
    models['net_ppg'] = LSTM_net(
        n_inputs=80, n_outputs=36, n_layers=2, hidden_size=128,
        fc_size=4096, dropout=0.2, bidirectional=True
    ).to(device)
    ppg_path = model_dir / 'msp_2_ppgmat_0_best.pth'
    models['net_ppg'].load_state_dict(torch.load(str(ppg_path)))

    # F0MATモデル
    models['net_f0mat'] = LSTM_net(n_inputs=80+36, n_outputs=80).to(device)
    f0mat_path = model_dir / 'msp_2_f0mat_0_best.pth'
    models['net_f0mat'].load_state_dict(torch.load(str(f0mat_path)))

    # MCPモデル
    models['net_mcp'] = LSTM_net(n_inputs=80+36, n_outputs=80).to(device)
    mcp_path = model_dir / 'msp_2_mcp_0_best.pth'
    models['net_mcp'].load_state_dict(torch.load(str(mcp_path)))

    # CAPモデル
    models['net_cap'] = LSTM_net(n_inputs=80+36, n_outputs=2).to(device)
    cap_path = model_dir / 'msp_2_cap_0_best.pth'
    models['net_cap'].load_state_dict(torch.load(str(cap_path)))

    # 推論モードに設定
    for model in models.values():
        model.eval()

    return models


def process_folder(input_folder, output_base_folder, models, device):
    """
    フォルダ内のWAVファイルを処理して保存する。

    Parameters:
        input_folder (Path): 入力フォルダのパス
        output_base_folder (Path): 出力フォルダのベースパス
        models (dict): ロードされたモデルの辞書
        device: 使用するデバイス (GPU/CPU)
    """
    input_folder = Path(input_folder)
    output_base_folder = Path(output_base_folder)
    print(f"Processing folder: {input_folder}")
    print(list(input_folder.iterdir()))
    # .wavファイルを再帰的に検索
    wav_files = list(input_folder.rglob("*.wav"))

    if not wav_files:
        print(f"Warning: No WAV files found in {input_folder}")
        return

    print(f"Found {len(wav_files)} WAV files")

    for input_path in wav_files:
        # 相対パスを計算
        relative_path = input_path.relative_to(input_folder)
        output_path = output_base_folder / relative_path

        # 処理と保存
        process_and_save(input_path, output_path, models['net_ppg'],
                         models['net_f0mat'], models['net_mcp'],
                         models['net_cap'], device)


def main():
    """
    スクリプトのエントリーポイント。
    whisper_converted_v2とwhisper10から有声音声を生成する。

    音響解析は別途 acoustic_analysis.py を使用してください。
    """
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # モデルのロード
    print("Loading models...")
    models = load_models(device)

    # 実験設定
    base_dir = Path(__file__)

    # 処理するフォルダの設定
    folders_to_process = [
        (base_dir.joinpath("dataset", "whisper_converted_v2"),
         base_dir.joinpath("results", "generated", "whisper_converted_v2")),
        (base_dir.joinpath("dataset", "whisper10"),
         base_dir.joinpath("results", "generated", "whisper10")),
    ]

    # 有声音声の生成
    print("\n=== Generating voiced speech ===")
    for input_folder, output_folder in folders_to_process:
        print(f"\nProcessing: {input_folder}")
        process_folder(input_folder, output_folder, models, device)

    print("\n=== Speech synthesis complete ===")
    print(f"Generated audio saved in: {base_dir / 'results' / 'generated'}")
    print("\nTo calculate acoustic metrics with DTW alignment, run:")
    print("  python 20251205_create_whisper2voice/scripts/"
          "acoustic_analysis.py")


# スクリプトの実行
if __name__ == "__main__":
    main()
    