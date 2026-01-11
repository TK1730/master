import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

import argparse
import sys

from utils.logger import logger

# デバイスを指定する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PathConfig:

    def __init__(self, dataset_root: str, assets_root: str):
        """
        文字列として受け取ったパスを、Pathオブジェクトに変換して保存する
        Args:
            dataset_root (str): データセットのルートパス
            assets_root (str): アセットのルートパス
        """
        self.dataset_root = Path(dataset_root)
        self.assets_root = Path(assets_root)


class ResampleConfig:
    def __init__(self, in_dir: str, out_dir: str, sample_rate: int = 22050, dataset: str = ""):
        """
        Args:
            in_dir (str): 入力ディレクトリ
            out_dir (str): 出力ディレクトリ
            sample_rate (int, optional): サンプリングレート. Defaults to 22050.
        """
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.sample_rate = sample_rate
        # dataset name (optional): logical subfolder under dataset root
        self.dataset = dataset

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """辞書からインスタンスを生成する"""
        # allow an optional dataset name: if provided, interpret in_dir/out_dir
        # relative to dataset_path/dataset, otherwise relative to dataset_path
        dataset_name = data.get("dataset", "")

        data["in_dir"] = data["in_dir"]
        data["out_dir"] = dataset_path / data["out_dir"]
        # ensure dataset key is present for downstream use
        data["dataset"] = dataset_name

        return cls(**data)


class Preprocess_text_config:
    """データ前処理設定"""

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int = 10,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        """
        テキストデータの前処理設定
        Args:
            transcription_path (str): トランスクリプションファイルのパス
            cleaned_path (str): 前処理後のデータを保存するパス
            train_path (str): 学習用データリストの出力先パス
            val_path (str): 検証用データリストの出力先パス
            config_path (str): 設定ファイルのパス
            val_per_lang (int, optional): 言語ごとの検証データ数. Defaults to 10.
            max_val_total (int, optional): 最大検証データ数. Defaults to 10000.
            clean (bool, optional): データをクリーンアップするかどうか. Defaults to True.
        """
        self.transcription_path = Path(transcription_path)
        if cleaned_path == "" or cleaned_path is None:
            self.cleaned_path = self.transcription_path.with_name(
                self.transcription_path.name + ".cleaned"
            )
        else:
            self.cleaned_path = Path(cleaned_path)
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.config_path = Path(config_path)
        self.val_per_lang = val_per_lang
        self.max_val_total = max_val_total
        self.clean = clean

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """辞書からインスタンスを生成する"""
        data["transcription_path"] = dataset_path / data["transcription_path"]
        data["cleaned_path"] = (
            dataset_path / data["cleaned_path"]
            if data["cleaned_path"] != "" and data["cleaned_path"] is not None
            else ""
        )
        data["train_path"] = dataset_path / data["train_path"]
        data["val_path"] = dataset_path / data["val_path"]
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class TrainConfig:

    def __init__(
        self,
        config_path: str,
        env: dict[str, Any],
        model_dir: str,
        spec_cache: bool,
        keep_ckpts: int,
        num_workers: int = 0,
    ):
        """学習設定

        Args:
            config_path (str): configファイルのパス
            env (dict[str, Any]): 環境変数
            model_dir (str): モデルの保存先ディレクトリ
            spec_cache (bool): スペクトログラムのキャッシュを使用するかどうか
            keep_ckpts (int): 保存するチェックポイントの数
            num_workers (int, optional): データローディングに使用するワーカーの数. Defaults to 0.
        """
        self.config_path = Path(config_path)
        self.env = env
        self.model_dir = Path(model_dir)
        self.spec_cache = spec_cache
        self.keep_ckpts = keep_ckpts
        self.num_workers = num_workers

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """辞書からインスタンスを生成する"""
        # data["model_dir"] = dataset_path / data["model_dir"]
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Config:

    def __init__(self, config_path: str, path_config: PathConfig):
        """
        コンフィグクラスの初期化

        Args:
            config_path (str): コンフィグファイルのパス
            path_config (PathConfig): パス設定
        """
        if not Path(config_path).exists():
            shutil.copy(src="configs/default_config.yaml", dst=config_path)
            logger.info(
                f"A configuration file {config_path} has been generated based "
                f"on the default configuration file default_config.yaml."
            )
            logger.info(
                "Please do not modify default_config.yaml. Instead, modify "
                "config.yaml."
            )

        with open(config_path, encoding="utf-8") as file:
            yaml_config: dict[str, Any] = yaml.safe_load(file)
            model_name: str = yaml_config["model_name"]
            self.model_name: str = model_name
            dataset_path = path_config.dataset_root
            self.dataset_path: Path = dataset_path
            self.dataset_root: Path = path_config.dataset_root
            self.assets_root: Path = path_config.assets_root
            self.out_dir: Path = self.assets_root
            self.resample_config: ResampleConfig = ResampleConfig.from_dict(
                dataset_path=dataset_path,
                data=yaml_config["resample"],
            )
            self.preprocess_text_config: Preprocess_text_config = (
                Preprocess_text_config.from_dict(
                    dataset_path=dataset_path,
                    data=yaml_config["preprocess_text"],
                )
            )
            self.train_config: TrainConfig = TrainConfig.from_dict(
                dataset_path=dataset_path,
                data=yaml_config["train_ms"],
            )


# Load and initialize the configuration
def get_path_config() -> PathConfig:
    path_config_path = Path("configs/paths.yaml")
    if not path_config_path.exists():
        shutil.copy(src="configs/default_paths.yaml", dst=path_config_path)
        logger.info(
            f"A configuration file {path_config_path} has been generated "
            f"based on the default configuration file "
            f"configs/default_paths.yaml."
        )

    with open(path_config_path, encoding="utf-8") as file:
        path_config_dict: dict[str, str] = yaml.safe_load(file.read())
    return PathConfig(**path_config_dict)


def get_config() -> Config:
    path_config = get_path_config()
    try:
        config = Config("configs/config.yaml", path_config)
    except (TypeError, KeyError):
        logger.warning("Old config.yaml found. Replace it with default_config.yaml.")
        shutil.copy(src="configs/default_config.yaml", dst="config.yaml")
        config = Config("config.yaml", path_config)
    return config


class InteractivePrompter:
    """対話で引数を上書きするユーティリティクラス"""

    @classmethod
    def ask_str(cls, prompt: str, current: str) -> str:
        v = input(f"{prompt} [{current}]: ").strip()
        return current if v == "" else v

    @classmethod
    def ask_int(cls, prompt: str, current: int) -> int:
        v = input(f"{prompt} [{current}]: ").strip()
        if v == "":
            return int(current)
        try:
            return int(v)
        except ValueError:
            print("Invalid integer. using default")
            return int(current)

    @classmethod
    def ask_float(cls, prompt: str, current: float) -> float:
        v = input(f"{prompt} [{current}]: ").strip()
        if v == "":
            return float(current)
        try:
            return float(v)
        except ValueError:
            print("Invalid float. using default")
            return float(current)

    @classmethod
    def ask_bool(cls, prompt: str, current: bool) -> bool:
        v = input(f"{prompt} [{current}]: ").strip().lower()
        if v == "":
            return current
        if v in ["true", "1", "yes", "y", "True"]:
            return True
        elif v in ["false", "0", "no", "n", "False"]:
            return False
        else:
            print("Invalid boolean. using default")
            return current


class ResamplePrompter(InteractivePrompter):
    def prompt(self, args: argparse.Namespace) -> argparse.Namespace:
        # 非対話環境では何もしない
        if not sys.stdin.isatty():
            return args
        print(
            "値を入力してください。デフォルト値を使用する場合はEnterを押してください。"
        )
        args.dataset = InteractivePrompter.ask_str(
            "Dataset name", str(args.dataset)
        )
        args.sr = InteractivePrompter.ask_int("Sampling rate", args.sr)
        args.input_dir = InteractivePrompter.ask_str(
            "Input directory", str(args.input_dir)
        )
        args.output_dir = InteractivePrompter.ask_str(
            "Output directory", str(args.output_dir)
        )
        args.cpu_processes = InteractivePrompter.ask_int(
            "If you want to use single CPU process, enter 1; otherwise, enter",
            args.cpu_processes,
        )
        args.loudness_normalize = InteractivePrompter.ask_bool(
            "Loudness normalize audio", args.loudness_normalize
        )
        args.trim = InteractivePrompter.ask_bool("Trim silence", args.trim)
        return args


class TextPrompter(InteractivePrompter):
    def prompt(self, args: argparse.Namespace) -> argparse.Namespace:
        # 非対話環境では何もしない
        if not sys.stdin.isatty():
            return args
        print(
            "値を入力してください。デフォルト値を使用する場合はEnterを押してください。"
        )
        args.transcription_path = InteractivePrompter.ask_str(
            "Input transcription path", str(args.transcription_path)
        )
        args.cleaned_path = InteractivePrompter.ask_str(
            "Cleaned path", str(args.cleaned_path)
        )
        args.train_path = InteractivePrompter.ask_str(
            "Output train list path", str(args.train_path)
        )
        args.val_path = InteractivePrompter.ask_str(
            "Output val list path", str(args.val_path)
        )
        args.config_path = InteractivePrompter.ask_str(
            "Config path", str(args.config_path)
        )
        args.val_per_lang = InteractivePrompter.ask_int(
            "Number of validation data per SPEAKER, not per language "
            "(due to compatibility with the original code).",
            args.val_per_lang,
        )
        args.max_val_total = InteractivePrompter.ask_int(
            "Maximum number of validation data", args.max_val_total
        )
        args.use_jp_extra = InteractivePrompter.ask_bool(
            "Use Japanese extra mora list", args.use_jp_extra
        )
        args.yomi_error = InteractivePrompter.ask_bool(
            "Yomi error correction", args.yomi_error
        )
        args.correct_path = InteractivePrompter.ask_str(
            "Yomi correction path", str(args.correct_path)
        )
        return args


if __name__ == "__main__":
    config = get_config()
    print(config.model_name)
    print(config.resample_config.__dict__.keys())
    print(config.resample_config.__dict__.values())
    print(config.train_config.__dict__.keys())
