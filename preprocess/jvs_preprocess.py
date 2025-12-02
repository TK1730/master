import numpy as np
import librosa
import pyworld as pw
import soundfile as sf

from pathlib import Path
import pyloudnorm as pyln
from numpy.typing import NDArray

import utils.config as config
import utils.functions as functions

# JVSコーパスのデータセットパス
jvs_dataset_path = Path('dataset/jvs_ver1')
# 前処理後のデータセットパス
output_dir = Path('dataset/preprocessed/jvs_ver1')
# 音声ファイルを書き出す際のサンプリンレート
output_wav_sampling_rate = config.sr
# 前処理の対象外とする音声ファイルの最小の長さ 2sec以下は除外する
min_wav_length = config.sr * 2
# 音素辞書の生成
phonemedict = functions.generate_phoneme_dict("./preprocess/phoneme.txt")


class BlockSizeException(Exception):
    pass


def normalize_audio(
    data: NDArray,
    sr: int,
    loudness_target: float = -28.0
) -> NDArray:
    """ラウドネスノーマライゼーションによる音声の正規化

    Args:
        data (NDArray): 音声データ
        sr (int): サンプリングレート
        loudness_target (float, optional): 目標ラウドネス値. Defaults to -28.0.
    """
    DEFAULT_BLOCK_SIZE = 0.4  # ITU-R BS.1770-4 recommends 400ms
    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError as e:
        raise BlockSizeException(e)

    data = pyln.normalize.loudness(data, loudness, loudness_target)

    return data


def preprocess_using_transcripts(
    speaker_id: int,
    jvs_dataset_path: Path,
    filedir: str
):
    """ transcripts_utf8.txtを使用して、音声データを前処理する関数

    Args:
        speaker_id ( int ): 話者ID
        jvs_dataset_path ( Path ): JVSコーパスのデータセットパス
        filedir ( str ): 前処理したい音声データのフォルダ名
    """
    # 話者IDに"jvs"を付与 0から始まるため+1する
    speaker_id_jvs = f'jvs{speaker_id+1:03}'
    # transcripts_utf8.txtのパス
    transcripts_path = jvs_dataset_path.joinpath(
        speaker_id_jvs,
        filedir,
        'transcripts_utf8.txt'
    )
    assert transcripts_path.exists(), (
            f"transcripts_utf8.txt not found for speaker {speaker_id_jvs}")
    # 音声ディレクトリのパス
    wav_dir = jvs_dataset_path.joinpath(
        speaker_id_jvs,
        filedir,
        'wav24kHz16bit'
    )
    # labelディレクトリのパス
    label_dir = jvs_dataset_path.joinpath(
        speaker_id_jvs,
        filedir,
        'lab',
        'mon'
    )
    # 出力ディレクトリに入っているファイルを記述したテキストファイルのパス
    output_txt_path = output_dir.joinpath(
        filedir,
        speaker_id_jvs,
        'transcripts_utf8.txt'
    )
    # 出力されたファイル名を記述するリスト
    output_txt_list = []

    # transcripts_utf8.txtを読み込み
    with open(transcripts_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 各行を処理
        for line in lines:
            line_split = line.split(':')
            # 音声データ取得
            wav_file_name = f"{line_split[0]}.wav"
            load_wav_file = wav_dir.joinpath(wav_file_name)
            # ラベルデータ取得
            label_file_name = f"{line_split[0]}.lab"
            load_label_file = label_dir.joinpath(label_file_name)

            if load_wav_file.exists():
                # ######### wavファイルに関する処理 ##########
                #  wavファイルをサンプリングレートoutput_wav_sampling_rate[Hz]に変換する
                loaded_wav_file, _ = librosa.load(
                    load_wav_file,
                    sr=output_wav_sampling_rate
                )
                if (loaded_wav_file.shape[0] < min_wav_length):
                    print(f"excluded from preprocessing: "
                          f"speaker: {speaker_id_jvs} {filedir} "
                          f"{wav_file_name} (len: {loaded_wav_file.shape[0]})")
                    continue

                # ラウドネスノーマライゼーション
                loaded_wav_file = normalize_audio(
                    loaded_wav_file,
                    output_wav_sampling_rate,
                    loudness_target=config.lufs_mix
                )

                # stft
                D = librosa.stft(y=loaded_wav_file,
                                 n_fft=config.n_fft,
                                 hop_length=config.hop_length,
                                 win_length=config.win_length,
                                 center=False,
                                 pad_mode='reflect')

                sp, _ = librosa.magphase(D)
                msp = np.matmul(config.mel_filter, sp).T
                lmsp = functions.dynamic_range_compression(msp)

                # f0, cp, ap 取得
                f0, cp, ap = pw.wav2world(loaded_wav_file.astype(np.float64),
                                          config.sr,)
                # f0mat
                f0dif = f0.reshape(-1, 1)*config.f0_scale - config.mel_freqs
                f0mat = np.eye(
                    config.n_mels
                )[(f0dif**2).argmin(axis=1)].astype(np.float32)
                f0mat[:, 0] = 0
                # 非周期性指標
                cap = pw.code_aperiodicity(ap, fs=config.sr)

                # メルケプストラム
                mcp = pw.code_spectral_envelope(cp, config.sr, config.n_mels)

                # ラベル取得
                if load_label_file.exists():
                    ppg = np.zeros(msp.shape[0])
                    loaded_label_file = open(
                        load_label_file, 'r', encoding='utf-8'
                    )
                    for line_label in loaded_label_file.readlines():
                        labs = line_label.split(' ')
                        start = int(float(labs[0])*config.sr/config.hop_length)
                        ppg[start:] = phonemedict[labs[2].replace('\n', '')]
                    loaded_label_file.close()
                    ppgmat = np.eye(36)[
                        np.array(ppg).astype(np.uint8)].astype(np.float32)

                # ######### データ保存処理 ##########
                # ファイル名を出力リストに追加
                output_txt_list.append(f"{str(line)}")

                # wavファイルの出力
                if not output_dir.joinpath(
                    filedir, speaker_id_jvs, "wav"
                ).exists():
                    output_dir.joinpath(
                        filedir, speaker_id_jvs, "wav"
                    ).mkdir(parents=True, exist_ok=True)

                output_wav_path = output_dir.joinpath(
                    filedir, speaker_id_jvs, "wav", wav_file_name
                )  # 出力する音声ファイルのパス
                sf.write(
                    output_wav_path,
                    loaded_wav_file,
                    config.sr,
                    subtype="PCM_16"
                )  # 音声ファイルを書き出し

                # npyファイルの出力
                if not output_dir.joinpath(
                    filedir, speaker_id_jvs, "npy"
                ).exists():
                    output_dir.joinpath(
                        filedir, speaker_id_jvs, "npy"
                    ).mkdir(parents=True, exist_ok=True)

                output_npy_path = output_dir.joinpath(
                    filedir, speaker_id_jvs, "npy"
                )
                np.save(output_npy_path.joinpath(
                    wav_file_name.replace('.wav', '_msp.npy')
                ), lmsp)
                np.save(output_npy_path.joinpath(
                    wav_file_name.replace('.wav', '_f0.npy')
                ), f0)
                np.save(output_npy_path.joinpath(
                    wav_file_name.replace('.wav', '_f0mat.npy')
                ), f0mat)
                np.save(output_npy_path.joinpath(
                    wav_file_name.replace('.wav', '_mcp.npy')
                ), mcp)
                np.save(output_npy_path.joinpath(
                    wav_file_name.replace('.wav', '_cap.npy')
                ), cap)
                # labelファイルの出力
                if load_label_file.exists():
                    output_label_path = output_dir.joinpath(
                        filedir,
                        speaker_id_jvs,
                        "npy"
                    )
                    np.save(output_label_path.joinpath(
                        label_file_name.replace('.lab', '_ppg.npy')
                    ), ppg)
                    np.save(output_label_path.joinpath(
                        label_file_name.replace('.lab', '_ppgmat.npy')
                    ), ppgmat)

    # 出力されたファイル名をテキストファイルに書き出し
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for output_txt in output_txt_list:
            f.write(output_txt)


if __name__ == '__main__':
    # jvs001-jvs100までの話者IDを持つ音声データを前処理
    for speaker_id in range(100):
        preprocess_using_transcripts(speaker_id, jvs_dataset_path, 'nonpara30')
        preprocess_using_transcripts(speaker_id, jvs_dataset_path, 'whisper10')
        preprocess_using_transcripts(
            speaker_id,
            jvs_dataset_path,
            'nonpara30w_ver1'
        )
        preprocess_using_transcripts(
            speaker_id,
            jvs_dataset_path,
            'nonpara30w_ver2'
        )
        print(f"preprocess speaker {speaker_id+1} processed")

    print("finish")
