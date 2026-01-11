import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperProcessor


# wavファイル、話者id、テキスト(音素列)の3つを読み込むためのDatasetクラス
class AudioSpeakerTextLoader(Dataset):
    """
    1) 前処理によって作成されたtxtファイルに書かれたwavファイル、話者id、テキスト(音素列)の3つを読み込む
    2) テキストを正規化し整数へと変換
    3) wavファイルからスペクトログラムを計算
    """
    def __init__(self, dataset_txtfile_path):
        """
        dataset_txtfile_path : 前処理によって作成されたtxtファイルへのパス
        """
        # whisperの使用に合わせたスペクトログラムの計算のためのパラメータ
        self.sampling_rate = 22050
        self.filter_length = 1024  # N_FFT
        self.hop_length = 256
        self.win_length = 1024

        # 前処理によって作成されたtxtファイルの読み込み
        # 一行につき
        # wavファイルへのパス|話者id|音素列
        with open(dataset_txtfile_path, "r", encoding="utf-8") as f:
            self.wavfilepath_speakerid_text = [
                line.strip().split("|")
                for line in f
                if line.strip()
            ]
        # 各行をランダムにシャッフル
        random.seed(1234)
        random.shuffle(self.wavfilepath_speakerid_text)

        # whisperのプロセッサーを定義
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-small",
            language="Japanese",
            task="transcribe"
        )

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # filename, speaker_id and textを分離
        audiopath, sid = audiopath_sid_text
        wav, spec = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (wav, spec, sid)

    def get_audio(self, wavfile_path):
        # wavファイルの読み込み
        wav, sr = torchaudio.load(wavfile_path)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        # wavからspectrogramを計算
        pad_size = int((self.filter_length - self.hop_length) / 2)
        wav_padded = nn.functional.pad(
            wav,
            (pad_size, pad_size),
            mode='reflect'
        )
        spec = torchaudio.functional.spectrogram(
            waveform=wav_padded,
            pad=0,
            window=torch.hann_window(self.win_length),
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=1,
            normalized=False,
            center=False
        )
        spec = torch.squeeze(spec, 0)
        return wav_padded, spec

    def get_audio_whisper(self, wavfile_path):
        # 16kHzで読み込み
        wav, sr = torchaudio.load(wavfile_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # whisperのプロセッサーによるメルスペクトログラムを計算
        mel_spec = self.processor.feature_extractor(
            wav.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
        ).input_features[0]

        return mel_spec

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        line = self.wavfilepath_speakerid_text[index]
        wavfilepath, speakerid = line[0], line[1]
        wav, spec = self.get_audio(wavfilepath)
        mel_spec = self.get_audio_whisper(wavfilepath)
        speaker_id = self.get_sid(speakerid)
        return (wav, spec, mel_spec, speaker_id)

    def __len__(self):
        return len(self.wavfilepath_speakerid_text)


# AudioSpeakerTextLoaderの__getitem__により取得されたデータをバッチへと固める関数
def collate_fn(batch):
    # batch = [
    #     (wav, spec, mel_spec, speaker_id),
    #     (wav, spec, mel_spec, speaker_id),
    #     ....
    # ]
    max_wav_len = max([x[0].size(1) for x in batch])  # wavの最大の長さを算出
    max_spec_len = max([x[1].size(1) for x in batch])  # spectrogramの最大の長さを算出
    max_mel_len = max([x[2].size(1) for x in batch])  # melspectrogramの最大の長さを算出

    batch_size = len(batch)

    wav_lengths = torch.LongTensor(batch_size)
    spec_lengths = torch.LongTensor(batch_size)
    mel_lengths = torch.LongTensor(batch_size)
    speaker_id = torch.LongTensor(batch_size)

    wav_padded = torch.zeros(batch_size, 1, max_wav_len, dtype=torch.float32)
    spec_padded = torch.zeros(
        batch_size, batch[0][1].size(0), max_spec_len, dtype=torch.float32
    )
    mel_padded = torch.zeros(
        batch_size, batch[0][2].size(0), max_mel_len, dtype=torch.float32
    )

    # 左詰めで元のデータを上書きすることによりzero-paddingされたtensorを取得できる
    for i, (wav_row, spec_row, mel_row, speaker_id_row) in enumerate(batch):
        wav_padded[i, :, :wav_row.size(1)] = wav_row
        wav_lengths[i] = wav_row.size(1)

        spec_padded[i, :, :spec_row.size(1)] = spec_row
        spec_lengths[i] = spec_row.size(1)

        mel_padded[i, :, :mel_row.size(1)] = mel_row
        mel_lengths[i] = mel_row.size(1)

        speaker_id[i] = speaker_id_row

    return (
        wav_padded, wav_lengths,
        spec_padded, spec_lengths,
        speaker_id,
        mel_padded, mel_lengths
    )


# batch内の各tensorについて、start_indices[i]で指定されたindexから長さsegment_sizeの箇所を取り出す関数
# 学習時、スペクトログラムや音声波形について、時間軸に沿って指定した長さだけ切り取るのに用いる
def slice_segments(input_tensor, start_indices, segment_size):
    output_tensor = torch.zeros_like(input_tensor[:, ..., :segment_size])
    batch_size = input_tensor.size(0)
    for batch_index in range(batch_size):
        index_start = start_indices[batch_index]
        index_end = index_start + segment_size
        output_tensor[batch_index] = input_tensor[
            batch_index,
            ...,
            index_start:index_end
        ]
    return output_tensor


if __name__ == "__main__":
    loader = AudioSpeakerTextLoader("dataset/sample_train.txt")
    wav, spec, mel_spec, speaker_id = loader[0]
    print(wav.shape)
    print(spec.shape)
    print(mel_spec.shape)
    print(speaker_id.shape)

    print("\n--- Test collate_fn ---")
    data_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    for i, batch in enumerate(data_loader):
        wav_padded, wav_lengths, \
            spec_padded, spec_lengths, \
            speaker_ids, \
            mel_padded, mel_lengths = batch

        print(f"Batch {i}:")
        print(f"  wav_padded: {wav_padded.shape}")
        print(f"  wav_lengths: {wav_lengths}")
        print(f"  spec_padded: {spec_padded.shape}")
        print(f"  spec_lengths: {spec_lengths}")
        print(f"  speaker_ids: {speaker_ids.shape}")
        print(f"  mel_padded: {mel_padded.shape}")
        print(f"  mel_lengths: {mel_lengths}")
        break
