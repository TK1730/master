import librosa

# audio config
n_mels = 80
n_fft = 1024
n_mgc = 80  # mel-cepstrum order
hop_length = 256  # WARNING: this can't be changed.
win_length = 1024
sr = 22050
fmin = 0.0
fmax = 8000.0
silence_threshold = -55

# network config
dropout = 0.2
hidden_size = 256
num_layers = 4
output_size = 36

# train config
epochs = 1000
test_rate = 0.2
batch_size = 20
frame_length = 128
f0_scale = 10

# laudnes
# jvs_nonpara30 + throat
lufs_mix = -38.07440141136585
lufs_jvs = -27.657454012957633
lufs_throat = -54.773324721791454
lufs_nonpara30w = -21.540267870444595

# melfilter
mel_freqs = librosa.mel_frequencies(
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax,
    htk=False
).reshape(1, -1)
mel_filter = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    fmin=fmin,
    fmax=fmax,
    n_mels=n_mels,
    htk=False,
    norm='slaney'
)
