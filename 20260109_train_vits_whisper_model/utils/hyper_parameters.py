from distutils import config
import json
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


class HyperParametersTrain(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 1234
    epochs: int = 1000
    iterations: int = 200000
    learning_rate: float = 2e-4
    betas: list[float] = [0.8, 0.99]
    eps: float = 1e-9
    batch_size: int = 16
    bf16_run: bool = False
    fp16_run: bool = False
    lr_decay: float = 0.999875
    segment_size: int = 8192
    init_lr_ratio: float = 1.0
    warmup_epochs: int = 0
    accumulation_steps: int = 1
    c_mel: int = 45
    c_kl: float = 1.0
    c_commit: int = 100
    skip_optimizer: bool = False


class HyperParametersData(BaseModel):
    training_files: str = "Data/Dummy/train.list"
    validation_files: str = "Data/Dummy/val.list"
    max_wav_value: float = 32768.0
    sampling_rate: int = 22050
    filter_length: int = 1024  # FFTのサイズ
    hop_length: int = 256
    win_length: int = 1024
    use_mel: bool = False
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    add_blank: bool = True
    n_speakers: int = 1
    use_jp_extra: bool = False
    cleaned_text: bool = True
    spk2id: dict[str, int] = {"Dummy": 0}


class HyperParametersModel(BaseModel):
    use_spk_conditoned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_duration_discriminator: bool = False
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = [3, 7, 11]
    resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates: list[int] = [8, 8, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list = [16, 16, 4, 4]
    n_layers_q: int = 3
    use_spectral_norm: bool = False  # GANの識別器で使用
    gin_channels: int = 256


class HyperParameters(BaseModel):
    model_name: str = "Dummy"
    version: str = "2.0"
    train: HyperParametersTrain = HyperParametersTrain()
    data: HyperParametersData = HyperParametersData()
    model: HyperParametersModel = HyperParametersModel()

    # 以下は学習時にのみ動的に設定されるパラメータ (通常 config.yamlには存在しない)
    model_dir: Optional[str] = None
    speedp: bool = False  # 学習高速化モード
    repo_id: Optional[str] = None  # HuggingFace HubのリポジトリID

    # model_ 以下をPydantic の保護対象から除外する
    model_config = ConfigDict(protected_namespaces=())

    @staticmethod
    def load_from_json(json_path: Union[str, Path]) -> "HyperParameters":
        """JSONファイルからハイパーパラメータを読み込む

        Args:
            json_path (Union[str, Path]): JSONファイルのパス

        Returns:
            HyperParameters: 読み込まれたハイパーパラメータ
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return HyperParameters.model_validate_json(f.read())


if __name__ == "__main__":
    hps = HyperParameters.load_from_json("config.json")
    print(hps)
