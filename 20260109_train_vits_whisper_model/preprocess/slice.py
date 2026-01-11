import argparse
import shutil
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utils.config import get_path_config
from utils.logger import logger
from utils.stdout_wrapper import SAFE_STDOUT


def is_audio_file(file: Path) -> bool:
    """音声ファイルかどうかを判定する

    Args:
        file (Path): ファイルパス

    Returns:
        bool: 音声ファイルならTrue、そうでなければFalse
    """
    audio_extensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    return file.suffix.lower() in audio_extensions


def get_stamps(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    min_silence_dur_ms: int = 700,
    min_sec: float = 2,
    max_sec: float = 12,
):
    """音声ファイルから発話区間の開始と終了のタイムスタンプを取得する

    Args:
        vad_model (Any):
            発話区間検出(VAD)するためのモデル
            事前学習済みのモデルSilero VADモデルを使用することを想定
        utils (Any): VADモデルのユーティリティ関数群
        audio_file (Path): 音声ファイルのパス
        min_silence_dur_ms (int, optional):
            無音区間の最小持続時間（ミリ秒） Defaults to 700.
            このミリ秒数以上を無音とみなす
            小さくすると、音声がぶつ切りに小さくなりすぎ、
            大きくすると音声一つ一つが長くなりすぎる
        min_sec (float, optional):
            発話区間の最小長さ（秒） Defaults to 2.
            この秒数より小さい発話は無視される
        max_sec (float, optional):
            発話区間の最大長さ（秒） Defaults to 12.
            この秒数より大きい発話は無視される
    """
    (get_speech_timestamps, _, read_audio, *_) = utils
    sampling_rate = 16000  # 16kHzか8kHzのみ対応

    # ミリ秒に変換
    min_ms = int(min_sec * 1000)

    wav = read_audio(str(audio_file), sampling_rate=sampling_rate)
    get_speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=min_ms,
        max_speech_duration_ms=max_sec,
    )

    return get_speech_timestamps


def split_wav(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    target_dir: Path,
    min_sec: float = 2,
    max_sec: float = 12,
    min_silence_dur_ms: int = 700,
    time_suffix: bool = False,
) -> tuple[float, int]:
    """音声の無声部分を除去して.発話区間ごとに分割し、ファイルに保存する

    Args:
        vad_model (Any):
            発話区間検出(VAD)するためのモデル
            事前学習済みのモデルSilero VADモデルを使用することを想定
        utils (Any): VADモデルのユーティリティ関数群
        audio_file (Path): 音声ファイルのパス
        target_dir (Path): 出力先ディレクトリ
        min_sec (float, optional): 発話区間の最小長さ（秒） Defaults to 2.
        max_sec (float, optional): 発話区間の最大長さ（秒） Defaults to 12.
        min_silence_dur_ms (int, optional): 無音区間の最小持続時間（ミリ秒） Defaults to 700.
        time_suffix (bool, optional):
            出力ファイル名にタイムスタンプを付加するかどうか
            Defaults to False.
            付加する場合、ファイル名は「{basename}_{start}_{end}.wav」の形式になる
    """
    # ミリ秒単位で,音声の前後に無音を持たせる
    margin: int = 200
    speech_timestamps = get_stamps(
        vad_model=vad_model,
        utils=utils,
        audio_file=audio_file,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )
    # 音声ファイルを読み込む
    data, sr = sf.read(audio_file)
    # 音声全体の長さをミリ秒単位で取得
    total_ms = len(data) / sr * 1000
    # ファイル名を取得
    file_name = audio_file.stem

    total_time_ms: float = 0.0
    count = 0

    # タイムスタンプに従って分割し、ファイルに保存
    for i, ts in enumerate(speech_timestamps):
        start_ms = max(ts["start"] / 16 - margin, 0)
        end_ms = min(ts["end"] / 16 + margin, total_ms)

        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment = data[start_sample:end_sample]

        if time_suffix:
            file = f"{file_name}-{int(start_ms)}-{int(end_ms)}.wav"
        else:
            file = f"{file_name}_{i}.wav"

        sf.write(str(target_dir / file), segment, sr)
        total_time_ms += end_ms - start_ms
        count += 1

    return total_time_ms / 1000, count


def remove_silence(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    target_dir: Path,
    min_sec: float = 2,
    max_sec: float = 12,
    min_silence_dur_ms: int = 700,
) -> tuple[float, int]:
    """音声の無声部分を除去して.wavで保存する

    Args:
        vad_model (Any):
            発話区間検出(VAD)するためのモデル
            事前学習済みのモデルSilero VADモデルを使用することを想定
        utils (Any): VADモデルのユーティリティ関数群
        audio_file (Path): 音声ファイルのパス
        target_dir (Path): 出力先ディレクトリ
        min_sec (float, optional): 発話区間の最小長さ（秒） Defaults to 2.
        max_sec (float, optional): 発話区間の最大長さ（秒） Defaults to 12.
        min_silence_dur_ms (int, optional): 無音区間の最小持続時間（ミリ秒） Defaults to 700.
    """
    # ミリ秒単位で,音声の前後に無音を持たせる
    margin: int = 200
    speech_timestamps = get_stamps(
        vad_model=vad_model,
        utils=utils,
        audio_file=audio_file,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )
    # 音声ファイルを読み込む
    data, sr = sf.read(audio_file)
    # 音声全体の長さをミリ秒単位で取得
    total_ms = len(data) / sr * 1000
    # ファイル名を取得
    file_name = audio_file.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    total_time_ms: float = 0.0
    count = 0

    # タイムスタンプに従って分割し、ファイルに保存
    all_segments = []
    for i, ts in enumerate(speech_timestamps):
        start_ms = max(ts["start"] / 16 - margin, 0)
        end_ms = min(ts["end"] / 16 + margin, total_ms)

        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment = data[start_sample:end_sample]

        all_segments.append(segment)

        total_time_ms += end_ms - start_ms
        count += 1

    # リスト内の全セグメントを連結
    combined_segments = np.concatenate(all_segments)
    # 結果をファイルに保存
    file = f"{file_name}.wav"
    sf.write(str(target_dir / file), combined_segments, sr)

    return total_time_ms / 1000, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_sec", "-m", type=float, default=2, help="Minimum seconds of a slice"
    )
    parser.add_argument(
        "--max_sec", "-M", type=float, default=12, help="Maximum seconds of a slice"
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="inputs",
        help="Directory of input wav files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The result will be in Data/{model_name}/raw/ "
        "(if Data is dataset_root in configs/paths.yml)",
    )
    parser.add_argument(
        "--min_silence_dur_ms",
        "-s",
        type=int,
        default=700,
        help="Silence above this duration (ms) " "is considered as a split point.",
    )
    parser.add_argument(
        "--time_suffix",
        "-t",
        action="store_true",
        help="Make the filename end with -start_ms-end_ms when saving wav.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=3,
        help="Number of processes to use. Default 3 seems to be the best.",
    )
    args = parser.parse_args()

    path_config = get_path_config()
    dataset_root = path_config.dataset_root

    model_name = str(args.model_name)
    input_dir = Path(args.input_dir)
    output_dir = dataset_root / model_name / "raw"
    min_sec: float = args.min_sec
    max_sec: float = args.max_sec
    min_silence_dur_ms: int = args.min_silence_dur_ms
    time_suffix: bool = args.time_suffix
    num_processes: int = args.num_processes

    audio_files = [file for file in input_dir.rglob("*") if is_audio_file(file)]

    logger.info(f"Found {len(audio_files)} audio files.")
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists, " f"deleting...")
        shutil.rmtree(output_dir)

    # モデルをダウンロードしておく
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        onnx=True,
    )

    # Silero VADのモデルは、同じインスタンスで並列処理するとおかしくなるらしい
    # ワーカーごとにモデルをロードするようにするため、Queueを使って処理する
    def process_queue(
        q: Queue[Optional[Path]],
        result_queue: Queue[tuple[float, int]],
        error_queue: Queue[tuple[Path, Exception]],
    ):
        # logger.debug("Worker started.")
        vad_model, utils = torch.hub.load(
            repo_or_dir="litagin02/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True,
        )
        while True:
            file = q.get()
            if file is None:  # 終了シグナルを確認
                q.task_done()
                break
            try:
                rel_path = file.relative_to(input_dir)
                time_sec, count = split_wav(
                    vad_model=vad_model,
                    utils=utils,
                    audio_file=file,
                    target_dir=output_dir / rel_path.parent,
                    min_sec=min_sec,
                    max_sec=max_sec,
                    min_silence_dur_ms=min_silence_dur_ms,
                    time_suffix=time_suffix,
                )
                result_queue.put((time_sec, count))
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                error_queue.put((file, e))
                result_queue.put((0, 0))
            finally:
                q.task_done()

    q: Queue[Optional[Path]] = Queue()
    result_queue: Queue[tuple[float, int]] = Queue()
    error_queue: Queue[tuple[Path, Exception]] = Queue()

    # ファイル数が少ない場合は、ワーカー数をファイル数に合わせる
    num_processes = min(num_processes, len(audio_files))

    threads = [
        Thread(target=process_queue, args=(q, result_queue, error_queue))
        for _ in range(num_processes)
    ]
    for t in threads:
        t.start()

    pbar = tqdm(total=len(audio_files), file=SAFE_STDOUT, dynamic_ncols=True)
    for file in audio_files:
        q.put(file)

    # result_queueを監視し、要素が追加されるごとに結果を加算しプログレスバーを更新
    total_sec = 0
    total_count = 0
    for _ in range(len(audio_files)):
        time, count = result_queue.get()
        total_sec += time
        total_count += count
        pbar.update(1)

    # 全ての処理が終わるまで待つ
    q.join()

    # 終了シグナル None を送る
    for _ in range(num_processes):
        q.put(None)

    for t in threads:
        t.join()

    pbar.close()

    if not error_queue.empty():
        error_str = "Error slicing some files:"
        while not error_queue.empty():
            file, e = error_queue.get()
            error_str += f"\n{file}: {e}"
        raise RuntimeError(error_str)

    logger.info(
        f"Slice done! Total time: {total_sec / 60:.2f} min, " f"{total_count} files."
    )
