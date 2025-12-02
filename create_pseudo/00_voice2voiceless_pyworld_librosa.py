"""
疑似ささやき声生成用スクリプト 新しいタイプ
"""
import numpy as np
import librosa
import pyworld as pw
import cv2
import matplotlib.pyplot as plt
import soundfile as sf
import shutil
from pathlib import Path

dataset_path = Path('dataset/jvs_ver1')  # JVSコーパスのデータセットパス
voiced_path = 'nonpara30'  # 有声発話のディレクトリ名
voiceless_path = Path('whisper10')  # 無声発話のディレクトリ名
output_path = Path('nonpara30w_ver2')  # 結果の保存先のディレクトリ名

# 前処理の対象外とする音声ファイルの最小の長さ 2sec以下は除外する
sr = 22050
min_wav_length = sr * 2

high_pass = False


def pseudo_whisper_voice_processing(
        dataset_path: Path,
        voiced_dir: str,
        voiceless_dir: str,
        output_dir: str
):
    """有声発話から疑似的なささやき声をjvsコーパスから生成する

    Args:
        dataset_path ( Path )   : jvsコーパスのデータセットパス
        voiced_dir ( str )      : 有声発話のディレクトリ名
        voiceless_dir ( str )   : 無声発話のディレクトリ名
        output_dir ( str)       : 結果の保存先ディレクトリ名
    """

    for i in range(100):
        # 話者IDに"jvs"を付与 0から始まるため+1する
        speaker_id_jvs = f'jvs{i+1:03}'
        person = dataset_path.joinpath(speaker_id_jvs)
        # 有声発話のディレクトリパス
        voiced_wav_path = person.joinpath(voiced_dir, 'wav24kHz16bit')
        # 無声発話のディレクトリパス
        voiceless_wav_path = person.joinpath(voiceless_dir, 'wav24kHz16bit')
        # 保存先のディレクトリパス
        output_path = Path.joinpath(
            person, output_dir, 'wav24kHz16bit'
        )
        output_path.mkdir(parents=True, exist_ok=True)

        print(speaker_id_jvs+'>>>>>>>>>>>>>>>')
        # ######### 無声化処理 ##########
        # voiced
        for idx, load_wav_file in enumerate(voiced_wav_path.iterdir()):
            loaded_wav_file, sr = librosa.load(load_wav_file)
            if loaded_wav_file.shape[0] < min_wav_length:
                continue
            f0, cp, ap = pw.wav2world(loaded_wav_file.astype(np.float64), sr)
            if idx == 0:
                cp_all = cp.T
            else:
                cp_all = np.hstack((cp_all, cp.T))

        x = cp_all.max(axis=1)
        xmax = x.max()
        x /= xmax

        # voiceless
        for idx, load_wav_file in enumerate(voiceless_wav_path.iterdir()):
            loaded_wav_file, sr = librosa.load(load_wav_file)
            f0, cp, ap = pw.wav2world(loaded_wav_file.astype(np.float64), sr)
            D = librosa.stft(
                y=loaded_wav_file,
                n_fft=1024
            )  # hop_length = round(sr*0.005))
            sp, phase = librosa.magphase(D)

            if idx == 0:
                cp_all = cp.T
                sp_all = sp
            else:
                cp_all = np.hstack((cp_all, cp.T))
                sp_all = np.hstack((sp_all, sp))

        y = cp_all.max(axis=1)
        ymax = y.max()
        y /= ymax
        b = np.zeros_like(x)
        b = y/x

        # apply b
        for idx, load_wav_file in enumerate(voiced_wav_path.iterdir()):
            loaded_wav_file, sr = librosa.load(load_wav_file)
            f0, sp, ap = pw.wav2world(loaded_wav_file.astype(np.float64), sr)

            rsp = sp*b*ymax
            rsp.clip(0, None)

            rf0 = np.zeros_like(f0)
            rap = np.ones_like(ap)  # .copy().clip(0.5,None)
            # synthesize an utterance using the parameters
            rx = pw.synthesize(rf0, rsp, rap, sr)
            D = librosa.stft(y=rx, n_fft=1024)
            sp, phase = librosa.magphase(D)

            if idx == 0:
                rsp_all = sp
            else:
                rsp_all = np.hstack((rsp_all, sp))

        rb = np.mean(sp_all, axis=1) / np.mean(rsp_all, axis=1)

        plt.clf()
        fig = plt.figure()
        plt.plot(sp_all.max(axis=1), label='whisper')
        plt.plot(rsp_all.max(axis=1), label='pseudo')
        plt.plot(rb, label='R')
        plt.xlabel('Frequency bin (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid()
        # plt.close()
        fig.canvas.draw()
        im = np.array(fig.canvas.renderer.buffer_rgba())
        dst = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)

        cv2.imshow('b', dst)
        cv2.waitKey(1)

        for idx, load_wav_file in enumerate(voiced_wav_path.iterdir()):
            output_wav_path = Path.joinpath(output_path, load_wav_file.name)

            wav, sr = librosa.load(load_wav_file)
            f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr)

            rsp = sp*b*ymax
            rsp.clip(0, None)

            rf0 = np.zeros_like(f0)
            rap = np.ones_like(ap)  # .copy().clip(0.5,None)
            # synthesize an utterance using the parameters
            rx = pw.synthesize(rf0, rsp, rap, sr)

            D = librosa.stft(y=rx, n_fft=1024)
            sp, phase = librosa.magphase(D)

            rsp = (sp.T*rb).T

            rD = rsp * np.exp(1j*phase)  # 直交形式への変換はlibrosaの関数ないみたいなので、自分で計算する。
            rwav = librosa.istft(rD)

            sf.write(output_wav_path, rwav, sr, subtype="PCM_16")

            # # spectrogram 描画
            # # 無声発話
            # whisper_path = voiceless_wav_path.joinpath(load_wav_file.name)
            # if not whisper_path.exists():
            #     continue
            # whisper_wav, sr = librosa.load(whisper_path)
            # D = librosa.stft(
            #     y=whisper_wav,
            #     n_fft=1024
            # )  # , hop_length = round(sr*0.005))
            # whisper_sp, _ = librosa.magphase(D)
            # i
            # # 有声発話
            # D = librosa.stft(
            #     y=wav,
            #     n_fft=1024
            # )  # , hop_length = round(sr*0.005))
            # voice_sp, _ = librosa.magphase(D)
            # # 疑似無声発話
            # pseudo_sp = rsp

            # # 周波数方向に平均化
            # voice_sp_mean = voice_sp.mean(axis=1)
            # whisper_sp_mean = whisper_sp.mean(axis=1)
            # pseudo_sp_mean = pseudo_sp.mean(axis=1)

            # # 描画
            # plt.clf()
            # fig = plt.figure()
            # plt.plot(voice_sp_mean, label='voice')
            # plt.plot(whisper_sp_mean, label='whisper')
            # plt.plot(pseudo_sp_mean, label='pseudo')
            # plt.xlabel('Frequency bin (Hz)')
            # plt.ylabel('Magnitude')
            # plt.legend()
            # plt.grid()
            # plt.close()
            # plt.show()

        # ######### ラベルファイルとモノラベルのコピー処理 ##########
        voiced_dir_path = Path.joinpath(person, voiced_dir)  # 有声発話のディレクトリパス
        output_dir_path = Path.joinpath(person, output_dir)  # 保存先のディレクトリパス

        # ファイルをコピー
        shutil.copy2(
            Path.joinpath(voiced_dir_path, "transcripts_utf8.txt"),
            output_dir_path
        )
        # フォルダをコピー
        shutil.copytree(
            Path.joinpath(voiced_dir_path, "lab/mon"),
            output_dir_path.joinpath("lab/mon"),
            dirs_exist_ok=True
        )


pseudo_whisper_voice_processing(
    dataset_path,
    voiced_path,
    voiceless_path,
    output_path
)
print("Processing completed.")
