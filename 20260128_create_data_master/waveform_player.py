"""
音声波形表示と再生位置インジケータ付きプレイヤー

音声ファイルを読み込み、波形を表示しながら現在の再生位置を
リアルタイムで表示するシンプルなプログラム。
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


class WaveformPlayer:
    """音声波形表示と再生位置インジケータ機能を持つプレイヤークラス。"""

    def __init__(self, audio_path: str) -> None:
        """
        初期化処理。

        Args:
            audio_path: 音声ファイルのパス
        """
        self.audio_path = Path(audio_path)
        self.audio: Optional[np.ndarray] = None
        self.sr: int = 22050

    def load_audio(self) -> None:
        """音声ファイルを読み込み、正規化する。"""
        print(f"Loading audio: {self.audio_path}")
        self.audio, self.sr = librosa.load(str(self.audio_path), sr=None)

        # 音声を正規化 (最大振幅を1.0に)
        max_val = np.max(np.abs(self.audio))
        if max_val > 0:
            self.audio = self.audio / max_val
        print(f"Sample rate: {self.sr} Hz")
        print(f"Duration: {len(self.audio) / self.sr:.2f} seconds")
        print("Audio normalized.")

    def play(self) -> None:
        """音声を再生し、波形と再生位置を表示する。"""
        if self.audio is None:
            self.load_audio()

        # 波形表示の設定
        fig, ax = plt.subplots(figsize=(12, 4))
        duration = len(self.audio) / self.sr
        time_axis = np.linspace(0, duration, len(self.audio))

        # 波形をプロット
        ax.plot(time_axis, self.audio, color='steelblue', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform: {self.audio_path.name}')
        ax.set_xlim(0, duration)
        ax.set_ylim(-1, 1)

        # 再生位置インジケータ（赤い縦線）
        line, = ax.plot([0, 0], [-1, 1], color='red', linewidth=2)

        plt.tight_layout()
        plt.ion()
        plt.show()

        # 音声再生開始
        print("Starting playback...")
        sd.play(self.audio, samplerate=self.sr)
        start_time = time.time()

        # インジケータ更新ループ
        try:
            while sd.get_stream().active:
                elapsed = time.time() - start_time
                if elapsed > duration:
                    break
                line.set_xdata([elapsed, elapsed])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.03)
        except KeyboardInterrupt:
            sd.stop()
            print("\nPlayback stopped.")

        print("Playback finished.")
        plt.ioff()
        plt.show()


def main() -> None:
    """メイン関数。"""
    parser = argparse.ArgumentParser(
        description='音声波形表示と再生位置インジケータ付きプレイヤー'
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='再生する音声ファイルのパス'
    )
    args = parser.parse_args()

    player = WaveformPlayer(args.audio_file)
    player.play()


if __name__ == '__main__':
    main()
