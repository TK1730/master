import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import librosa
from numpy.typing import NDArray
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm
import yaml

# _project_root = Path(__file__).resolve().parents[1]
# if str(_project_root) not in sys.path:
#     sys.path.insert(0, str(_project_root))
from utils.config import get_config
from utils.logger import logger
from utils.stdout_wrapper import SAFE_STDOUT

# ラウドネスノーマライゼーションの処理する時間幅
DEFAULT_BLOCK_SIZE = 0.4  # ITU-R BS.1770-4 recommends 400ms


class BlockSizeException(Exception):
    pass


def normalize_audio(data: NDArray, sr: int, loudness_target: float = -28.0) -> NDArray:
    """ラウドネスノーマライゼーションによる音声の正規化

    Args:
        data (NDArray): 音声データ
        sr (int): サンプリングレート
        loudness_target (float, optional): 目標ラウドネス値. Defaults to -28.0.
    """
    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError as e:
        raise BlockSizeException(e)

    data = pyln.normalize.loudness(data, loudness, loudness_target)

    return data


def resample(
    file: Path,
    input_dir: Path,
    outputdir: Path,
    target_sr: int,
    normalize: bool,
    trim: bool,
):
    """
    fileを読み込んで、target_srならwavファイルに変換して、
    output_dirの中に、input_dirからの相対パスを保つように保存する

    Args:
        file (Path): wavファイル
        input_dir (Path): 入力ディレクトリ
        outputdir (Path): 出力ディレクトリ
        target_sr (int): 目標サンプリングレート
        normalize (bool): 正規化フラグ
        trim (bool): トリミングフラグ
    """
    try:
        # librosaが読めるファイルかチェック
        # wav以外にもmp3やoggやflacなども読める
        # Load without resampling first (sr=None) and then explicitly resample
        # so we guarantee the output uses `target_sr`.
        wav: NDArray[Any]
        sr: int
        try:
            # Load directly at target_sr so librosa will resample on load
            # if necessary. This avoids calling librosa.resample explicitly.
            wav, sr = librosa.load(file, sr=int(target_sr))
            wav = wav / max(abs(wav).max(), 1e-9) * 0.9  # 正規化

        except Exception as e:
            logger.warning(f"Cannot load or resample file, so skip it: {file} ({e})")
            return
        if normalize:
            try:
                wav = normalize_audio(wav, sr)
            except BlockSizeException:
                print("")
                logger.info(
                    f"Skip normalize due to less than {DEFAULT_BLOCK_SIZE} "
                    f"seconds audio: {file}"
                )

        if trim:
            # 先頭と末尾の無音部分をトリミング
            wav, _ = librosa.effects.trim(wav, top_db=-55)  # -55dB以下を無音とみなす

        # 音声の長さが2秒以下の場合はスキップ
        if len(wav) / sr <= 2.0:
            logger.info(f"Skip resample due to less than 2 seconds audio: {file}")
            return
        else:
            wav = wav / max(abs(wav).max(), 1e-9)  # 再度正規化
            relative_path = file.relative_to(input_dir)
            # ここで拡張子が.wav以外でも.wavに置き換えられる
            output_path = outputdir / relative_path.with_suffix(".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # ファイルを書き出し
            sf.write(output_path, wav, target_sr, subtype="PCM_16")

    except Exception as e:
        # Catch any unexpected error for this file and skip it, but log details
        logger.warning(f"Error processing {file}, skipping: {e}")
        return


if __name__ == "__main__":
    from utils.config import ResamplePrompter

    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sample_rate,
        help="sampling rate",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=getattr(config.resample_config, "dataset", ""),
        help="logical dataset name (optional)",
    )
    parser.add_argument(
        "--cpu_processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    parser.add_argument(
        "--loudness_normalize",
        action="store_true",
        default=False,
        help="loudness normalize audio",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        default=False,
        help="trim silence (start and end only)",
    )
    args = parser.parse_args()
    prompter = ResamplePrompter()
    args = prompter.prompt(args)

    if args.cpu_processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes: int = args.cpu_processes

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logger.info(f"Resampling {input_dir} to {output_dir}")
    sr = int(args.sr)
    normalize: bool = args.loudness_normalize
    trim: bool = args.trim
    # 出力先ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    # resample設定を保存
    args_dict = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_dict[k] = str(v)
        else:
            args_dict[k] = v
    # yaml出力先ディレクトリを作成
    output_yaml = config.dataset_path.joinpath("config_yamls")
    output_yaml.mkdir(parents=True, exist_ok=True)
    # Save dataset.yaml under output_dir/<dataset>/dataset.yaml if dataset provided
    if args.dataset:
        print(f"Saving resample config under dataset directory: {args.dataset}")
        dataset_config_path = output_yaml.joinpath(f"{args.dataset}.yaml")
        with open(dataset_config_path, "w", encoding="utf-8") as f:
            yaml.dump(args_dict, f, sort_keys=False, allow_unicode=True)

    else:
        # fallback: save to output_dir.yaml (legacy behavior)
        with open(
            output_yaml.joinpath(output_dir.with_suffix(".yaml")
                                 ), "w", encoding="utf-8") as f:
            yaml.dump(args_dict, f, sort_keys=False, allow_unicode=True)

    logger.info(f"Saved resample config to {dataset_config_path}")

    # 後でlibrosaに読ませて有効な音声ファイルかチェックするので、全てのファイルを取得
    original_files = [f for f in input_dir.rglob("*") if f.is_file()]

    if len(original_files) == 0:
        logger.error(f"No files found in {input_dir}")
        raise ValueError(f"No files found in {input_dir}")

    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(
                resample,
                file,
                input_dir,
                output_dir,
                sr,
                normalize,
                trim,
            )
            for file in original_files
            if ".wav" in file.suffix.lower()
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        ):
            pass

    logger.info("Resampling Done!")
