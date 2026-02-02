"""
音声波形GUIアプリケーション

音声ファイルを読み込み、波形を表示し、任意の区間を選択して
画像として保存できるGUIアプリケーション。
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from matplotlib.patches import Rectangle
import numpy as np
from scipy import signal
import sounddevice as sd


class WaveformGUI:
    """音声波形表示・区間選択・保存機能を持つGUIアプリケーション。"""

    def __init__(self, root: tk.Tk) -> None:
        """
        初期化処理。

        Args:
            root: tkinterのルートウィンドウ
        """
        self.root = root
        self.root.title("Waveform Viewer")
        self.root.geometry("1200x700")

        # 音声データ
        self.audio: Optional[np.ndarray] = None
        self.sr: int = 22050
        self.audio_path: Optional[Path] = None

        # 2つ目の音声データ（DTW用）
        self.audio2: Optional[np.ndarray] = None
        self.sr2: int = 22050
        self.audio2_path: Optional[Path] = None

        # DTWアライメント
        self.dtw_path: Optional[np.ndarray] = None  # DTWパス
        self.aligned_audio2: Optional[np.ndarray] = None  # アライメントされた音声2
        self.dtw_enabled: bool = False  # DTWモード有効化

        # 選択範囲
        self.selection_start: Optional[float] = None
        self.selection_end: Optional[float] = None
        self.rect: Optional[Rectangle] = None
        self.is_selecting: bool = False

        # 再生位置
        self.playback_line1 = None  # 再生位置を示す赤い縦線（ax1）
        self.playback_line2 = None  # 再生位置を示す赤い縦線（ax2）
        self.playback_start_time: Optional[float] = None  # 再生開始時刻
        self.playback_offset: float = 0.0  # 再生開始位置（秒）
        self.is_playing: bool = False  # 再生中フラグ

        # UI構築
        self._create_widgets()

    def _create_widgets(self) -> None:
        """ウィジェットを作成する。"""
        # ツールバーフレーム
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=5)

        # ボタン
        ttk.Button(
            toolbar_frame, text="ファイル1を開く", command=self._open_file
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="クリア1", command=self._clear_file1
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="ファイル2を開く", command=self._open_file2
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="クリア2", command=self._clear_file2
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="DTWアライメント", command=self._compute_dtw
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame,
            text="元の波形⇔アライメント後",
            command=self._toggle_dtw_display
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        ttk.Button(
            toolbar_frame, text="再生", command=self._play_audio
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="選択範囲を再生", command=self._play_selection
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="停止", command=self._stop_audio
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        ttk.Button(
            toolbar_frame, text="選択範囲を画像保存", command=self._save_selection
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar_frame, text="全体を画像保存", command=self._save_full
        ).pack(side=tk.LEFT, padx=2)

        # 情報ラベル
        self.info_label = ttk.Label(toolbar_frame, text="ファイルを開いてください")
        self.info_label.pack(side=tk.RIGHT, padx=10)

        # 選択範囲ラベル
        self.selection_label = ttk.Label(toolbar_frame, text="選択: なし")
        self.selection_label.pack(side=tk.RIGHT, padx=10)

        # Matplotlibキャンバス（2つのサブプロット）
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # ナビゲーションツールバー
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, nav_frame)
        self.toolbar.update()

        # マウスイベント
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def _open_file(self) -> None:
        """ファイルダイアログで音声ファイルを開く。"""
        file_path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._load_audio(file_path)

    def _load_audio(self, file_path: str) -> None:
        """
        音声ファイルを読み込む。

        Args:
            file_path: 音声ファイルのパス
        """
        try:
            self.audio_path = Path(file_path)
            self.audio, self.sr = librosa.load(file_path, sr=None)

            # 正規化
            max_val = np.max(np.abs(self.audio))
            if max_val > 0:
                self.audio = self.audio / max_val

            self._plot_waveform()
            duration = len(self.audio) / self.sr
            self.info_label.config(
                text=f"{self.audio_path.name} | {self.sr}Hz | {duration:.2f}s"
            )
            self.selection_start = None
            self.selection_end = None
            self._update_selection_label()

        except Exception as e:
            messagebox.showerror("エラー", f"ファイルの読み込みに失敗しました:\n{e}")

    def _open_file2(self) -> None:
        """2つ目の音声ファイルを開く（DTW用）。"""
        file_path = filedialog.askopenfilename(
            title="2つ目の音声ファイルを選択",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._load_audio2(file_path)

    def _load_audio2(self, file_path: str) -> None:
        """
        2つ目の音声ファイルを読み込む。

        Args:
            file_path: 音声ファイルのパス
        """
        try:
            self.audio2_path = Path(file_path)
            self.audio2, self.sr2 = librosa.load(file_path, sr=None)

            # 正規化
            max_val = np.max(np.abs(self.audio2))
            if max_val > 0:
                self.audio2 = self.audio2 / max_val

            duration = len(self.audio2) / self.sr2
            messagebox.showinfo(
                "読み込み完了",
                f"ファイル2を読み込みました:\n{self.audio2_path.name}\n"
                f"{self.sr2}Hz | {duration:.2f}s"
            )

        except Exception as e:
            messagebox.showerror("エラー", f"ファイルの読み込みに失敗しました:\n{e}")

    def _compute_dtw(self) -> None:
        """DTWアライメントを計算する。"""
        if self.audio is None:
            messagebox.showwarning("警告", "ファイル1を開いてください")
            return
        if self.audio2 is None:
            messagebox.showwarning("警告", "ファイル2を開いてください")
            return

        try:
            # MFCCを計算（DTWの特徴量として使用）
            mfcc1 = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=self.audio2, sr=self.sr2, n_mfcc=13)

            # DTWを計算
            D, wp = librosa.sequence.dtw(mfcc1, mfcc2, metric='euclidean')
            self.dtw_path = wp

            # DTWパスに基づいて音声2をアライメント
            # 音声2のサンプルをファイル1の長さに合わせて伸縮
            hop_length = 512
            aligned_indices = []

            # フレームごとのアライメントをサンプルレベルに変換
            for i in range(len(self.audio)):
                # 現在のサンプルがどのフレームに対応するか
                frame1 = min(i // hop_length, mfcc1.shape[1] - 1)

                # DTWパスから対応するフレーム2を見つける
                matching_frames = wp[wp[:, 0] == frame1]
                if len(matching_frames) > 0:
                    frame2 = matching_frames[0, 1]
                    sample2 = frame2 * hop_length
                    aligned_indices.append(sample2)
                else:
                    aligned_indices.append(0)

            # 補間してアライメントされた音声2を作成
            aligned_indices = np.array(aligned_indices)
            aligned_indices = np.clip(aligned_indices, 0, len(self.audio2) - 1)
            self.aligned_audio2 = self.audio2[aligned_indices.astype(int)]

            self.dtw_enabled = True
            self._plot_waveform()

            messagebox.showinfo(
                "完了",
                f"DTWアライメントが完了しました\nDTWコスト: {D[-1, -1]:.2f}"
            )

        except Exception as e:
            messagebox.showerror("エラー", f"DTW計算に失敗しました:\n{e}")

    def _toggle_dtw_display(self) -> None:
        """DTWアライメント表示を切り替える。"""
        if self.aligned_audio2 is None:
            messagebox.showwarning("警告", "先にDTWアライメントを実行してください")
            return

        self.dtw_enabled = not self.dtw_enabled
        self._plot_waveform()

    def _clear_file1(self) -> None:
        """ファイル1をクリアする。"""
        self.audio = None
        self.audio_path = None
        self.selection_start = None
        self.selection_end = None
        self.dtw_path = None
        self.aligned_audio2 = None
        self.dtw_enabled = False
        self._plot_waveform()
        self.info_label.config(text="ファイル1がクリアされました")
        self._update_selection_label()

    def _clear_file2(self) -> None:
        """ファイル2をクリアする。"""
        self.audio2 = None
        self.audio2_path = None
        self.dtw_path = None
        self.aligned_audio2 = None
        self.dtw_enabled = False
        self._plot_waveform()
        messagebox.showinfo("完了", "ファイル2がクリアされました")

    def _plot_waveform(self) -> None:
        """波形を2つのサブプロットにプロットする。"""
        # 上段（ax1）: ファイル1
        self.ax1.clear()
        if self.audio is not None:
            duration1 = len(self.audio) / self.sr
            time_axis1 = np.linspace(0, duration1, len(self.audio))

            self.ax1.plot(
                time_axis1,
                self.audio,
                color='steelblue',
                linewidth=0.8,
                alpha=0.8
            )
            self.ax1.set_title(
                f'Audio 1: {self.audio_path.name}',
                fontsize=10
            )
            self.ax1.set_ylabel('Amplitude')
            self.ax1.set_ylim(-1.1, 1.1)
            self.ax1.grid(True, alpha=0.3)
        else:
            self.ax1.set_title('ファイル1を開いてください', fontsize=10)
            self.ax1.set_ylabel('Amplitude')

        # 下段（ax2）: ファイル2
        self.ax2.clear()
        if self.audio2 is not None and self.audio is not None:
            duration1 = len(self.audio) / self.sr
            time_axis1 = np.linspace(0, duration1, len(self.audio))

            # DTWアライメント後で、アライメント表示モードの場合
            if self.aligned_audio2 is not None and self.dtw_enabled:
                # アライメント後の音声2を表示
                self.ax2.plot(
                    time_axis1,
                    self.aligned_audio2,
                    color='orange',
                    linewidth=0.8,
                    alpha=0.8
                )
                mode_str = "(DTW Aligned)"
            else:
                # 元の音声2を長さを合わせて表示
                if len(self.audio2) != len(self.audio):
                    audio2_resampled = signal.resample(
                        self.audio2, len(self.audio)
                    )
                else:
                    audio2_resampled = self.audio2

                self.ax2.plot(
                    time_axis1,
                    audio2_resampled,
                    color='orange',
                    linewidth=0.8,
                    alpha=0.8
                )
                mode_str = "(Original)"

            self.ax2.set_title(
                f'Audio 2: {self.audio2_path.name} {mode_str}',
                fontsize=10
            )
            self.ax2.set_ylabel('Amplitude')
            self.ax2.set_ylim(-1.1, 1.1)
            self.ax2.grid(True, alpha=0.3)
        else:
            self.ax2.set_title('ファイル2を開いてください', fontsize=10)
            self.ax2.set_ylabel('Amplitude')

        self.ax2.set_xlabel('Time (s)')

        # 選択範囲と再生位置線をリセット
        self.rect = None
        self.playback_line1 = None
        self.playback_line2 = None

        self.fig.tight_layout()
        self.canvas.draw()

    def _on_press(self, event) -> None:
        """マウスボタン押下イベント。"""
        if event.inaxes != self.ax1 or self.audio is None:
            return
        if event.button == 1:  # 左クリック
            self.is_selecting = True
            self.selection_start = event.xdata
            self.selection_end = event.xdata
            self._update_selection_rect()

    def _on_motion(self, event) -> None:
        """マウス移動イベント。"""
        if not self.is_selecting or event.inaxes != self.ax1:
            return
        self.selection_end = event.xdata
        self._update_selection_rect()

    def _on_release(self, event) -> None:
        """マウスボタン解放イベント。"""
        if not self.is_selecting:
            return
        self.is_selecting = False
        if event.inaxes == self.ax1 and event.xdata is not None:
            self.selection_end = event.xdata

        # 開始と終了を正しい順序に
        if self.selection_start and self.selection_end:
            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = (
                    self.selection_end, self.selection_start
                )
        self._update_selection_label()

    def _update_selection_rect(self) -> None:
        """選択範囲の矩形を更新する。"""
        if self.rect:
            self.rect.remove()

        if self.selection_start is not None and self.selection_end is not None:
            x = min(self.selection_start, self.selection_end)
            width = abs(self.selection_end - self.selection_start)
            self.rect = Rectangle(
                (x, -1.1), width, 2.2,
                facecolor='yellow', alpha=0.3, edgecolor='orange'
            )
            self.ax1.add_patch(self.rect)
        self.canvas.draw()

    def _update_selection_label(self) -> None:
        """選択範囲ラベルを更新する。"""
        if self.selection_start is not None and self.selection_end is not None:
            duration = abs(self.selection_end - self.selection_start)
            self.selection_label.config(
                text=f"選択: {self.selection_start:.2f}s - "
                     f"{self.selection_end:.2f}s ({duration:.2f}s)"
            )
        else:
            self.selection_label.config(text="選択: なし")

    def _play_audio(self) -> None:
        """音声全体を再生する。"""
        if self.audio is None:
            messagebox.showwarning("警告", "音声ファイルを開いてください")
            return
        sd.stop()
        sd.play(self.audio, samplerate=self.sr)

        # 再生位置の追跡を開始
        self.playback_offset = 0.0
        self.playback_start_time = sd.get_stream().time
        self.is_playing = True
        self._update_playback_position()

    def _play_selection(self) -> None:
        """選択範囲を再生する。"""
        if self.audio is None:
            messagebox.showwarning("警告", "音声ファイルを開いてください")
            return
        if self.selection_start is None or self.selection_end is None:
            messagebox.showwarning("警告", "範囲を選択してください")
            return

        start_sample = int(self.selection_start * self.sr)
        end_sample = int(self.selection_end * self.sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio), end_sample)

        sd.stop()
        sd.play(self.audio[start_sample:end_sample], samplerate=self.sr)

        # 再生位置の追跡を開始
        self.playback_offset = self.selection_start
        self.playback_start_time = sd.get_stream().time
        self.is_playing = True
        self._update_playback_position()

    def _stop_audio(self) -> None:
        """再生を停止する。"""
        sd.stop()
        self.is_playing = False

        # 再生位置線を削除
        if self.playback_line1:
            self.playback_line1.remove()
            self.playback_line1 = None
        if self.playback_line2:
            self.playback_line2.remove()
            self.playback_line2 = None
        if self.playback_line1 is None and self.playback_line2 is None:
            self.canvas.draw()

    def _update_playback_position(self) -> None:
        """再生位置を更新する（タイマーで定期的に呼び出される）。"""
        if not self.is_playing:
            return

        # 再生が終了しているかチェック
        if not sd.get_stream().active:
            self.is_playing = False
            if self.playback_line1:
                self.playback_line1.remove()
                self.playback_line1 = None
            if self.playback_line2:
                self.playback_line2.remove()
                self.playback_line2 = None
            self.canvas.draw()
            return

        # 現在の再生位置を計算
        current_time = sd.get_stream().time
        elapsed_time = current_time - self.playback_start_time
        current_position = self.playback_offset + elapsed_time

        # 赤い縦線を更新（両方のサブプロット）
        if self.playback_line1:
            self.playback_line1.remove()
        if self.playback_line2:
            self.playback_line2.remove()

        self.playback_line1 = self.ax1.axvline(
            x=current_position,
            color='red',
            linewidth=2,
            alpha=0.8
        )
        self.playback_line2 = self.ax2.axvline(
            x=current_position,
            color='red',
            linewidth=2,
            alpha=0.8
        )
        self.canvas.draw()

        # 次の更新をスケジュール（50ms後）
        if self.is_playing:
            self.root.after(50, self._update_playback_position)

    def _save_selection(self) -> None:
        """選択範囲の波形を画像として保存する。"""
        if self.audio is None:
            messagebox.showwarning("警告", "音声ファイルを開いてください")
            return
        if self.selection_start is None or self.selection_end is None:
            messagebox.showwarning("警告", "範囲を選択してください")
            return

        file_path = filedialog.asksaveasfilename(
            title="画像を保存",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._save_waveform_image(
                file_path,
                self.selection_start,
                self.selection_end
            )

    def _save_full(self) -> None:
        """全体の波形を画像として保存する。"""
        if self.audio is None:
            messagebox.showwarning("警告", "音声ファイルを開いてください")
            return

        file_path = filedialog.asksaveasfilename(
            title="画像を保存",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            duration = len(self.audio) / self.sr
            self._save_waveform_image(file_path, 0, duration)

    def _save_waveform_image(
        self,
        file_path: str,
        start_time: float,
        end_time: float
    ) -> None:
        """
        指定範囲の波形を画像として保存する。

        Args:
            file_path: 保存先パス
            start_time: 開始時間(秒)
            end_time: 終了時間(秒)
        """
        try:
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(self.audio), end_sample)

            segment = self.audio[start_sample:end_sample]
            time_axis = np.linspace(start_time, end_time, len(segment))

            # 新しいFigureで保存
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time_axis, segment, color='steelblue', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{self.audio_path.name} [{start_time:.2f}s - {end_time:.2f}s]')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(file_path, dpi=150)
            plt.close(fig)

            messagebox.showinfo("完了", f"画像を保存しました:\n{file_path}")

        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")


def main() -> None:
    """メイン関数。"""
    root = tk.Tk()
    WaveformGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
