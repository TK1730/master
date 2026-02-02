import logging
import os
import random
import sys
from typing import Any

import torch
import torch.utils.data
from tqdm import tqdm

from utils.config import get_config
from utils.logger import logger
from utils.mel_processing import wav_to_mel, wav_to_spec
from utils.task import load_wav_to_torch, load_filepaths_and_text
from utils.hyper_parameters import HyperParametersData
from nlp import cleaned_text_to_sequence

# matplotlibのDEBUGログを抑制
logging.getLogger('matplotlib').setLevel(logging.WARNING)


config = get_config()


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) 音声, 話者id, textのペアをロード
    2) テキストを正規化し、整数のシーケンスに変換
    3) 音声ファイルからスペクトログラムを計算

    args:
        audiopaths_sid_text (str): ファイルパス|話者ID|言語|テキスト のリストファイル
        hparams (object): ハイパーパラメータ
    """

    def __init__(self, audiopaths_sid_text: str, hparams: HyperParametersData):
        """データローダーの初期化

        Args:
            audiopaths_sid_text (str): データセットのリストファイルパス
            hparams (HyperParametersData): ハイパーパラメータオブジェクト
        """
        # データセットのリストファイルをロード
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)

        # 音声関連のパラメータを設定
        self.max_wav_value = hparams.max_wav_value  # 音声波形の最大値（正規化用）
        self.sampling_rate = hparams.sampling_rate  # サンプリングレート
        self.filter_length = hparams.filter_length  # FFTのフィルタ長
        self.hop_length = hparams.hop_length  # STFTのホップ長
        self.win_length = hparams.win_length  # STFTのウィンドウ長

        # 話者関連のパラメータを設定
        self.spk_map = hparams.spk2id  # 話者名から話者IDへのマッピング
        self.use_jp_extra = hparams.use_jp_extra  # 日本語拡張機能の使用有無

        # ハイパーパラメータの参照を保持
        self.hparams = hparams

        # メルスペクトログラムの使用設定
        self.use_mel = getattr(hparams, "use_mel", False)
        if self.use_mel:
            # メルスペクトログラムのチャンネル数を設定（デフォルト: 80）
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        # テキストのクリーニング済みフラグ
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        # テキスト長のフィルタリング設定
        self.add_blank = hparams.add_blank  # 音素間にブランクトークンを挿入するか
        self.min_text_len = getattr(hparams, "min_text_len", 1)  # 最小テキスト長
        self.max_text_len = getattr(hparams, "max_text_len", 384)  # 最大テキスト長

        # データセットをシャッフル（再現性のためシードを固定）
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)

        # データセットのフィルタリング処理を実行
        self._filter()

    def _filter(self):
        """データセットのフィルタリングとスペクトログラム長の計算

        各データサンプルの音素、トーン、単語-音素マッピングを解析し、
        スペクトログラムの長さを推定してバケッティングに使用します。

        計算式:
            wav_length ≈ file_size / (wav_channels * Bytes per dim)
                       = file_size / (1 * 2)
            spec_length = wav_length // hop_length
        """
        audiopaths_sid_text_new = []
        lengths = []  # スペクトログラムの長さを格納
        skipped = 0   # スキップされたサンプル数

        logger.info("Init dataset...")
        # 各データサンプルを処理
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text, file=sys.stdout, dynamic_ncols=True
        ):
            audiopath = f"{_id}"

            # テキスト長のフィルタリング（現在はコメントアウト）
            # if self.min_text_len <= len(phones) and
            #    len(phones) <= self.max_text_len:

            # スペース区切りの文字列をリストに変換
            phones = phones.split(" ")  # 音素列
            tone = [int(i) for i in tone.split(" ")]  # トーン列
            word2ph = [int(i) for i in word2ph.split(" ")]  # 単語-音素マッピング

            # 処理済みデータを新しいリストに追加
            audiopaths_sid_text_new.append(
                [audiopath, spk, language, text, phones, tone, word2ph]
            )

            # ファイルサイズからスペクトログラムの長さを推定
            # ファイルサイズ / (チャンネル数 * 2バイト) / ホップ長
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))

        # フィルタリング結果をログ出力
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )

        # フィルタリング済みデータで上書き
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        """音声、テキスト、話者情報のペアを取得

        Args:
            audiopath_sid_text (list): [音声パス, 話者ID, 言語,
                テキスト, 音素, トーン, 単語-音素マッピング]

        Returns:
            tuple: (音素, スペクトログラム, 波形, 話者ID, トーン, 言語)
        """
        # 入力データをアンパック
        (
            audiopath,  # 音声ファイルのパス
            sid,        # 話者ID
            language,   # 言語情報
            text,       # 元のテキスト
            phones,     # 音素列
            tone,       # トーン列
            word2ph,    # 単語-音素マッピング
        ) = audiopath_sid_text

        # テキスト情報をテンソルに変換
        phones, tone, language = self.get_text(phones, tone, language)

        # 音声ファイルからスペクトログラムと波形を取得
        spec, wav = self.get_audio(audiopath)

        # 話者IDを整数テンソルに変換
        sid = torch.LongTensor([int(self.spk_map[sid])])

        # 日本語拡張機能の使用有無に関わらず同じデータを返す
        if self.use_jp_extra:
            return (phones, spec, wav, sid, tone, language)
        else:
            return (
                phones,
                spec,
                wav,
                sid,
                tone,
                language,
            )

    def get_audio(self, filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        """音声ファイルからスペクトログラムと波形を取得

        キャッシュがあればロードし、なければ音声ファイルから計算します。
        計算したスペクトログラムはキャッシュとして保存されます。

        Args:
            filename (str): 音声ファイルのパス

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                (スペクトログラム, 正規化された波形)

        Raises:
            ValueError: サンプリングレートが期待値と異なる場合
        """
        # 音声ファイルをロード
        audio, sampling_rate = load_wav_to_torch(filename)

        # サンプリングレートの検証
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{filename} {sampling_rate} SR doesn't match target "
                f"{self.sampling_rate} SR"
            )

        # 波形の正規化（-1.0 〜 1.0 の範囲にスケーリング）
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 0:
            audio_norm = audio / max_abs
        else:
            audio_norm = audio

        # スペクトログラムのキャッシュファイル名を生成
        spec_filename = filename.replace(".wav", ".spec.pt")

        # メルスペクトログラムを使う場合はファイル名を変更
        if self.use_mel:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")

        # キャッシュからスペクトログラムをロード
        try:
            spec = torch.load(spec_filename)
        except Exception:
            # キャッシュがない場合は音声から計算
            if self.use_mel:
                # メルスペクトログラムを計算
                spec = wav_to_mel(
                    audio,
                    n_fft=self.filter_length,
                    num_mels=self.n_mel_channels,
                    sampling_rate=self.sampling_rate,
                    hop_size=self.hop_length,
                    win_size=self.win_length,
                    fmin=self.hparams.mel_fmin,
                    fmax=self.hparams.mel_fmax,
                    center=False,
                    norm=True,
                )
            else:
                # 線形スペクトログラムを計算
                spec = wav_to_spec(
                    audio,
                    n_fft=self.filter_length,
                    sample_rate=self.sampling_rate,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    center=False,
                )
            # バッチ次元を削除
            spec = torch.squeeze(spec, 0)

        # スペクトログラムキャッシュが有効な場合は保存
        if config.train_config.spec_cache:
            torch.save(spec, spec_filename)

        return spec, audio_norm

    def get_text(
        self,
        phone: list[int],
        tone: list[int],
        language_str: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """テキストを整数シーケンスに変換

        Args:
            phone (list[int]): 音素のリスト
            tone (list[int]): トーンのリスト
            language_str (str): 言語の文字列(例: "JP", "EN")

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                音素、トーン、言語の整数シーケンス（テンソル形式）
        """
        # テキストのクリーニングと整数シーケンスへの変換
        phone, tone, language = cleaned_text_to_sequence(
            phone, tone, language_str
        )

        # ブランクトークンの挿入（音素間に0を挿入）
        if self.add_blank:
            phone = intersperse(phone, 0)
            tone = intersperse(tone, 0)
            language = intersperse(language, 0)

        # リストをテンソルに変換
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)

        return phone, tone, language

    def get_sid(self, sid):
        """話者IDをテンソルに変換

        Args:
            sid: 話者ID(文字列または整数)

        Returns:
            torch.Tensor: 話者IDのテンソル
        """
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        """指定されたインデックスのデータサンプルを取得

        Args:
            index (int): データサンプルのインデックス

        Returns:
            tuple: (音素, スペクトログラム, 波形, 話者ID, トーン, 言語)
        """
        return self.get_audio_text_speaker_pair(
            self.audiopaths_sid_text[index]
        )

    def __len__(self):
        """データセットのサイズを取得

        Returns:
            int: データセット内のサンプル数
        """
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """バッチデータのコレート処理を行うクラス

    モデルの入力とターゲットをゼロパディングし、バッチ内の全サンプルを
    同じ長さに揃えます。これにより、効率的なバッチ処理が可能になります。
    """

    def __init__(self, return_ids=False, use_jp_extra=False):
        """コレートの初期化

        Args:
            return_ids (bool): IDを返すかどうか(デフォルト: False)
            use_jp_extra (bool): 日本語拡張機能を使用するか(デフォルト: False)
        """
        self.return_ids = return_ids
        self.use_jp_extra = use_jp_extra

    def __call__(self, batch):
        """バッチデータのコレート処理

        正規化されたテキスト、音声、話者IDからトレーニングバッチを作成します。
        全てのシーケンスを最大長に合わせてゼロパディングします。

        Args:
            batch (list): データローダーから取得したバッチ
                各要素: [音素, スペクトログラム, 波形, 話者ID, トーン, 言語]

        Returns:
            tuple: パディング済みのテンソルのタプル
                (text_padded, text_lengths, spec_padded, spec_lengths,
                 wav_padded, wav_lengths, sid, tone_padded, language_padded)
        """
        # スペクトログラムの長さでソート（降順）
        # 長いシーケンスから処理することで効率的なパディングが可能
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0,
            descending=True
        )

        # バッチ内の最大長を取得
        max_text_len = max([len(x[0]) for x in batch])  # テキストの最大長
        max_spec_len = max([x[1].size(1) for x in batch])  # スペクトログラムの最大長
        max_wav_len = max([x[2].size(1) for x in batch])  # 波形の最大長

        # 長さを格納するテンソルを初期化
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        # パディング済みデータを格納するテンソルを初期化
        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)

        spec_padded = torch.FloatTensor(
            len(batch), batch[0][1].size(0), max_spec_len
        )
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        # 全てのテンソルを0で初期化
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        # ソート済みのインデックスで各サンプルを処理
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # テキスト（音素）のパディング
            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            # スペクトログラムのパディング
            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            # 波形のパディング
            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            # 話者IDの設定
            sid[i] = row[3]

            # トーンのパディング
            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            # 言語情報のパディング
            language = row[5]
            language_padded[i, : language.size(0)] = language

        # パディング済みのバッチデータを返す
        # 日本語拡張機能の有無に関わらず同じデータを返す
        if self.use_jp_extra:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
            )
        else:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
            )


def intersperse(lst: list[Any], item: Any) -> list[Any]:
    """
    リストの要素の間に特定のアイテムを挿入する

    Args:
        lst (list[Any]): 元のリスト
        item (Any): 挿入するアイテム

    Returns:
        list[Any]: 新しいリスト
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.hyper_parameters import HyperParameters

    # 1. ハイパーパラメータのロード
    # 実行ディレクトリからの相対パス
    config_path = "configs/config.json"
    hps = HyperParameters.load_from_json(config_path)

    # 2. データセットの初期化
    # 検証用データセットを使用
    dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)

    # 3. データの取得
    index = 0
    # get_audio_text_speaker_pair が内部で呼ばれる
    phones, spec, wav, sid, tone, language = dataset[index]

    logger.info(f"Loaded sample at index {index}")
    logger.info(f"Wav shape: {wav.shape}")
    logger.info(f"Spec shape: {spec.shape}")

    # 4. wav_to_mel を使用してメルスペクトログラムを計算
    # wav はすでに 正規化されている (1, T)
    mel = wav_to_mel(
        wav,
        n_fft=hps.data.filter_length,
        num_mels=hps.data.n_mel_channels,
        sampling_rate=hps.data.sampling_rate,
        hop_size=hps.data.hop_length,
        win_size=hps.data.win_length,
        fmin=hps.data.mel_fmin,
        fmax=hps.data.mel_fmax,
        center=False,
        norm=True,
    )

    logger.info(f"Mel : max={mel.max()}, min={mel.min()}")

    # 5. 描画と保存
    mel_np = mel.squeeze(0).cpu().numpy()
    mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min())
    import librosa.display

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_np,
        sr=hps.data.sampling_rate,
        hop_length=hps.data.hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=hps.data.mel_fmin,
        fmax=hps.data.mel_fmax,
        cmap="magma",
        vmin=mel_np.min(),
        vmax=mel_np.max()
    )
    plt.colorbar(format="%.2f dB")
    plt.title(f"Mel Spectrogram (index: {index})")
    plt.tight_layout()

    save_path = "dataset_mel_test.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved mel spectrogram to {save_path}")
