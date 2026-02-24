import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np

wav_path = "dataset/nonpara30/jvs001/wav/BASIC5000_0235.wav"

# 音声ファイルの読み込み
wav, sr = librosa.load(wav_path, sr=22050)

# wavの正規化 (-1 ~ 1)
wav = wav / np.max(np.abs(wav))

# DIOによる基本周波数の推定
f0, timeaxis = pw.dio(wav.astype(np.float64), sr)

# スペクトル包絡の推定
sp = pw.cheaptrick(wav.astype(np.float64), f0, timeaxis, sr)

# メルケプストラムの計算
mgc = pw.code_spectral_envelope(sp, sr, 24)
print(f"メルケプストラム形状: {mgc.shape}")
print(mgc.min())
print(mgc.max())

# 可視化パラメータ
hop_length = int(sr * 0.005)
num_frames = mgc.shape[0]
num_orders = mgc.shape[1]
duration = num_frames * hop_length / sr

# より見やすい図の作成
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# imshowを使用して明示的に軸範囲を設定
mesh = ax.imshow(
    mgc.T,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    extent=[0, duration, 0, num_orders],
    interpolation='bilinear'
)

# 軸ラベルとタイトルを設定
ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mel-Cepstral Order', fontsize=12, fontweight='bold')
ax.set_title('Mel-Cepstrum (PyWorld)', fontsize=14, fontweight='bold', pad=15)

# カラーバーを追加
cbar = fig.colorbar(mesh, ax=ax, format="%.2f")
cbar.set_label('Amplitude', fontsize=11, fontweight='bold')

# グリッドを追加（見やすくするため）
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
