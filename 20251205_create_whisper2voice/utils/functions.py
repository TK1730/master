import librosa
import pyworld
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import utils.config as config
from scipy import interpolate
from pathlib import Path
import librosa.display
import pyloudnorm as pyln
import re
import unicodedata

# 音声
def wav2msp(file_path, sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft, fmin=config.fmin, fmax=config.fmax, n_mels=config.n_mels, htk=False, norm='slaney', pad_mode='reflect', power=1):
    """
    Load audio to convert to msp
    """
    wav,sr = librosa.load(file_path,sr=config.sr)
    D = librosa.stft(y=wav, n_fft = config.n_fft, hop_length = config.hop_length, win_length = config.win_length, pad_mode='reflect').T
    sp,phase = librosa.magphase(D)
    msp = np.matmul(sp,config.mel_filter)
    return msp

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    convert logscale
    """
    return np.log(x.clip(clip_val,None))

def dynamic_range_decompression(x, C=1):
    """
    convert linear
    """
    return np.exp(x)

def interp1d(f0, kind='slinear'):
    ndim = f0.ndim
    if len(f0) != f0.size:
        raise RuntimeError("1d array is only supported")
    
    continuous_f0 = f0.flatten()
    nonzero_indices = np.where(continuous_f0 > 0)[0]

    if len(nonzero_indices) <= 0:
        return f0

    continuous_f0[0] = continuous_f0[nonzero_indices[0]]
    continuous_f0[-1] = continuous_f0[nonzero_indices[-1]]

    nonzero_indices = np.where(continuous_f0 > 0)[0]
    interp_func = interpolate.interp1d(
        nonzero_indices, continuous_f0[continuous_f0 > 0], kind=kind
    )

    zero_indices = np.where(continuous_f0 <= 0)[0]
    continuous_f0[zero_indices] = interp_func(zero_indices)

    if ndim == 2:
        return continuous_f0[:, None]
    return continuous_f0

mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax, htk=False).reshape(1,-1)
mel_filter = librosa.filters.mel(sr=config.sr, n_fft = config.n_fft, fmin= config.fmin, fmax= config.fmax, n_mels = config.n_mels, htk = False, norm='slaney').T


# # グラフ化
def msp2graph(msp, label='msp'):
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = librosa.display.specshow(msp, sr=config.sr, hop_length=config.hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="viridis")
    fig.colorbar(mesh, ax=ax, format="%+2.f dB")
    ax.set_title(label)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [Hz]")
    plt.show()
    mpl.pyplot.close()
    
def f02graph(f0, label='f0'):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(f0, linewidth=2, label=label)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [Hz]")
    # ax.legend()
    plt.show()
    mpl.pyplot.close()

def ap2graph(ap, label='ap', sr=config.sr, hop_length=config.hop_length):
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = librosa.display.specshow(ap.T, sr=sr, hop_length = hop_length, x_axis="time", y_axis="linear", ax=ax)
    ax.set_title("Aperiodicity")
    fig.colorbar(mesh, ax=ax, format="%+2.f dB")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(label)
    plt.tight_layout()
    plt.show()
    mpl.pyplot.close()

def sp2graph(sp, label='cp', sr=config.sr, hop_length=config.hop_length):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.set_xlabel("Time [sec]")
    log_sp = librosa.power_to_db(np.abs(sp), ref=np.max)
    mesh = librosa.display.specshow(log_sp.T, sr=sr, hop_length=config.hop_length, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(mesh, ax=ax, format="%+2.f dB")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(label)
    plt.tight_layout()
    plt.show()
    mpl.pyplot.close()

def best_model(folder):
    """
    学習済みモデルの結果がいいものを出力
    """
    path = Path(folder)
    if path.is_dir() == True:
        file = path.glob('*.npy')
        for i,data in enumerate(file):
            # npを読み込み
            model = np.load(data)
            if i == 0:
                best_model = model
                fname = data
                index = i
            # 比較
            if model < best_model:
                best_model = model
                fname = data
                index = i
        print(index)
        best = str(fname).replace('.npy', '.pth')
        # 結果がいいファイルパスを返す
        return best, index

    else:
        print('pathが通っていません')

def TrainDic(folder):
    """
    学習時に使用したデータをlistで返す
    """
    path = Path(folder)
    if path.is_dir() == True:
        file = path.glob('*')

def generate_phoneme_dict(path='./phoneme.txt'):
    """ Generate a phoneme dictionary from a text file.

    Args:
        path (str, optional): filepath. Defaults to './phoneme.txt'.

    Returns:
        dict : A dictionary mapping phonemes to their indices.
    """
    phonemelist = []

    f = open(path)                     #フォンテキストを開く
    for phoneme in f.readlines():
        phonemelist = phoneme.replace("'","").split(', ')
    f.close()

    phonemedict = {p: phonemelist.index(p) for p in phonemelist}    # 辞書型にしている
    phonemedict['a:'] = phonemedict['a'] # adhoc
    phonemedict['i:'] = phonemedict['i'] # adhoc
    phonemedict['u:'] = phonemedict['u'] # adhoc
    phonemedict['e:'] = phonemedict['e'] # adhoc
    phonemedict['o:'] = phonemedict['o'] # adhoc
    phonemedict['A'] = phonemedict['a'] # adhoc
    phonemedict['I'] = phonemedict['i'] # adhoc
    phonemedict['U'] = phonemedict['u'] # adhoc
    phonemedict['E'] = phonemedict['e'] # adhoc
    phonemedict['O'] = phonemedict['o'] # adhoc
    phonemedict['pau'] = phonemedict['sil'] # adhoc
    phonemedict['silB'] = phonemedict['sil'] # adhoc
    phonemedict['silE'] = phonemedict['sil'] # adhoc
    phonemedict['q'] = phonemedict['sil']
    phonemedict['sp'] = phonemedict['sil'] # adhoc
    phonemedict['cl'] = phonemedict['sil']

    print(phonemedict)

    return phonemedict

def loudness_average(file):
    wav, sr = librosa.load(file, sr=config.sr)
    meter = pyln.Meter(config.sr)
    lufs_src = meter.integrated_loudness(wav)
    return lufs_src

def loudness_normalize(wav, lufs_average, load=True):
    if load:
        wav, sr = librosa.load(wav, sr=config.sr)
    meter = pyln.Meter(config.sr)
    lufs_src = meter.integrated_loudness(wav)
    new_wav = pyln.normalize.loudness(wav, lufs_src, lufs_average)
    return new_wav


# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja よりコピペ
# encoding: utf8


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

def max(x, axis):
    result = np.zeros_like(x)
    index = np.argmax(x, axis=axis)
    for i, num in enumerate(index):
        result[i][num] = 1

    return result