"""
疑似ささやき声生成用スクリプト 古いタイプ
"""

import numpy as np
import librosa
import pyworld as pw
import os
import soundfile as sf
import shutil
from scipy import signal

dataset_path = './dataset/jvs_ver1/'
voiced_dir = '/nonpara30/'
voiced_wav_path = voiced_dir + 'wav24kHz16bit/'
voiceless_wav_path = '/whisper10/wav24kHz16bit/'
output_dir = '/nonpara30w_ver1/'
result_wav_path = output_dir + 'wav24kHz16bit/'
ouput_path = './dataset/jvs_ver1/'

high_pass = False

hp_freq = 240
hp_id = 40


def lpf(wav, sr, fcutoff):
    """ローパスフィルター"""
    nyq = sr / 2.0
    b, a = signal.butter(1, fcutoff/nyq, btype='high')
    wav = signal.filtfilt(b, a, wav)
    return wav.copy()


sr = 22050
nyq = sr / 2.0
od, wn = signal.buttord(240/nyq, 1, 1, 12)
b, a = signal.butter(od, wn, btype='high')


for person in os.listdir(dataset_path):
    if os.path.isdir(dataset_path + person):
        print(person+'>>>>>>>>>>>>>>>')

        # voiced
        sall = np.zeros(513).reshape(513, 1)
        for f in os.listdir(dataset_path + person+voiced_wav_path):
            file_path = dataset_path + person + voiced_wav_path + f
            wav, sr = librosa.load(file_path)
            f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr)
            sall = np.hstack((sall, sp.T))
        x = sall[:, 1:].mean(axis=1)
        xmax = x.max()
        x /= xmax
        sall = np.zeros(513).reshape(513, 1)


# voicelessf:/rabosemi/voiceless2voice/voiceless2voice/dataset/jvs_ver1/jvs_ver1/
        for f in os.listdir(dataset_path+person+voiceless_wav_path):
            file_path = dataset_path + person + voiceless_wav_path + f
            wav, sr = librosa.load(file_path)
            f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr)
            sall = np.hstack((sall, sp.T))

        y = sall[:, 1:].mean(axis=1)
        ymax = y.max()
        y /= ymax
        b = np.zeros_like(x)
        b[x > 0] = y[x > 0] / x[x > 0]
        b[b < 3] = 0.0001
        b[:hp_id] = 0.0001

        # apply b
        out_path = ouput_path + person + result_wav_path
        os.makedirs(out_path, exist_ok=True)
        for f in os.listdir(dataset_path+person+voiced_wav_path):
            file_path = dataset_path + person + voiced_wav_path + f
            outfile_path = out_path + f

            wav, sr = librosa.load(file_path)
            f0, sp, ap = pw.wav2world(wav.astype(np.float64), sr)

            rsp = (sp*b)
            rf0 = np.zeros_like(f0)
            rap = np.ones_like(ap)
            rx = pw.synthesize(rf0, rsp, rap, sr)
            if high_pass:
                rx = signal.filtfilt(b, a, rx)
            if rx.max() > 1.0:
                rx /= rx.max()

            sf.write(outfile_path, rx, sr, subtype="PCM_16")
        # ######## ラベルファイルとモノラベルのコピー処理 ##########
        voiced_dir_path = dataset_path + person + voiced_dir  # 有声発話のディレクトリパス
        output_dir_path = ouput_path + person + output_dir  # 保存先のディレクトリパス

        # ファイルをコピー
        shutil.copy2(
            voiced_dir_path + "/transcripts_utf8.txt",
            output_dir_path
        )
        # フォルダをコピー
        shutil.copytree(
            voiced_dir_path + "/lab/mon",
            output_dir_path + "/lab/mon",
            dirs_exist_ok=True
        )

print("finish")
