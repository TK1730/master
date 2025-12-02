"""
北見工業大学で収録された音声「throat_microphone_dataset_v2」を前処理するスクリプト
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import traceback

import utils.config as config
import utils.functions as functions

# throat_microphoneのデータセットパス
THROAT_DATASET_PATH = Path('dataset/throat_microphone_dataset_v2')
# 出力用ディレクトリ
OUTPUT_DIR = Path('./dataset/throat_microphone_preprocessed')
# 音声ファイルを書き出す際のサンプリンレート 22050hz
OUTPUT_WAV_SAMPLING_RATE = config.sr
# 前処理の対象外とする音声ファイルの最小の長さ 0.5sec以下は除外する
MIN_WAV_LENGTH = config.sr * 0.5
# 音素辞書の生成
PHONEMEDICT = functions.generate_phoneme_dict("./preprocess/phoneme.txt")


def preprocess_throat_data(speaker_id: int, throat_dataset_path: Path):
    """ throat_microphoneの音声データを前処理する関数
    Args:
        speaker_id (int): 話者ID
        throat_dataset_path (Path): throat_microphoneのデータセットパス
    """
    # 話者ID 40から始まるため+1する
    speaker_id_throat = str(f'{speaker_id:03}')
    # 音声ディレクトリのパス
    wav_dir = throat_dataset_path.joinpath(speaker_id_throat)
    # dy.wavのパスを取得
    for wav_file_name in wav_dir.glob('*dy.wav'):
        load_wav_file = wav_file_name
        # ラベルデータ取得
        load_label_file = load_wav_file.with_suffix('.lab')
        label_file_name = load_label_file.name
        if load_wav_file.exists():
            # wavファイルに関する処理
            # wavファイルをサンプリングレートOUTPUT_WAV_SAMPLING_RATE[Hz]に変換する
            loaded_wav_file, _ = librosa.load(
                load_wav_file,
                sr=OUTPUT_WAV_SAMPLING_RATE
            )

            # 音声ファイルの長さがMIN_WAV_LENGTHより短い場合は前処理の対象外とする
            if loaded_wav_file.shape[0] < MIN_WAV_LENGTH:
                print(f"excluded from preprocessing: speaker: "
                      f"{speaker_id_throat} "
                      f"{wav_file_name} "
                      f"(len: {loaded_wav_file.shape[0]})")
                continue

            # ラウドネスノーマライゼーション
            loaded_wav_file = functions.loudness_normalize(
                loaded_wav_file,
                config.lufs_mix,
                load=False
            )

            # stft
            D = librosa.stft(
                y=loaded_wav_file,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
                pad_mode='reflect'
            ).T  # 短時間フーリエ変換
            sp, _ = librosa.magphase(D)
            msp = np.matmul(sp, config.mel_filter.T)
            lmsp = functions.dynamic_range_compression(msp)

            try:
                # ラベル取得
                if load_label_file.exists():
                    # 音素ラベル処理
                    ppg = np.zeros(msp.shape[0])
                    loaded_label_file = open(
                        load_label_file,
                        'r',
                        encoding='utf-8'
                    )
                    for line_label in loaded_label_file.readlines():
                        labs = line_label.split(' ')
                        start = int(float(labs[0])*config.sr/config.hop_length)
                        ppg[start:] = PHONEMEDICT[labs[2].replace('\n', '')]
                    loaded_label_file.close()
                    ppgmat = np.eye(36)[
                        np.array(ppg).astype(np.uint8)
                    ].astype(np.float32)

                # wavファイルの出力
                if not OUTPUT_DIR.joinpath(speaker_id_throat).exists():
                    OUTPUT_DIR.joinpath(speaker_id_throat, "wav").mkdir(
                        parents=True,
                        exist_ok=True
                    )
                output_wav_path = OUTPUT_DIR.joinpath(
                    speaker_id_throat,
                    "wav",
                    wav_file_name.name
                )
                sf.write(output_wav_path, loaded_wav_file, config.sr, subtype="PCM_16")

                # npyファイルの出力
                if not OUTPUT_DIR.joinpath(speaker_id_throat, "npy").exists():
                    OUTPUT_DIR.joinpath(speaker_id_throat, "npy").mkdir(
                        parents=True,
                        exist_ok=True
                    )
                output_npy_path = OUTPUT_DIR.joinpath(speaker_id_throat, "npy")
                np.save(
                    output_npy_path.joinpath(wav_file_name.name.replace('.wav','_msp.npy')),
                    lmsp
                )

                # labelファイルの出力
                if load_label_file.exists():
                    output_label_path = OUTPUT_DIR.joinpath(speaker_id_throat, "npy")
                    np.save(
                        output_label_path.joinpath(label_file_name.replace('.lab','_ppg.npy')),
                        ppg
                    )
                    np.save(
                        output_label_path.joinpath(label_file_name.replace('.lab','_ppgmat.npy')),
                        ppgmat
                    )

            except Exception as e:
                print(f"Error processing {wav_file_name}: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    for id in range(41, 80+1):
        preprocess_throat_data(id, THROAT_DATASET_PATH)
        print(f"preprocess finished for speaker {id}")

    print("finish")
