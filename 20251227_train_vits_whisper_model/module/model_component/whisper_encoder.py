from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import WhisperModel
from safetensors.torch import load_file

try:
    from .text_encoder import MultiHeadAttention
except ImportError:
    from module.model_component.text_encoder import MultiHeadAttention


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int(((kernel_size - 1)*dilation)/2)


class ResNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResNetBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)

            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x


class WhisperEncoder(nn.Module):
    """
    OpenAIのWhisperモデルを使用して、音声データから音響特徴量を抽出するエンコーダークラスです。
    """
    def __init__(
        self,
        out_channels: int = 192,
        model_name: str = "openai/whisper-small",
        device: str = "cpu",
        weight_path: str | Path | None = None,
    ) -> None:
        """
        WhisperEncoderを初期化し、指定されたモデルをロードします。

        Args:
            out_channels (int): 出力チャネル数。 mとlogsの出力チャネル数。
            model_name (str): Hugging Faceのモデル名またはローカルパス。
            device (str): モデルを配置するデバイス (例: "cpu", "cuda")。
            weight_path (str | Path | None): ロードする追加の重みファイルパス。デフォルトはNone。
        """
        super().__init__()
        self.out_channels = out_channels
        self.device = device
        print(f"Loading model: {model_name} on {device}...")

        # Load only the encoder
        full_model = WhisperModel.from_pretrained(model_name)
        self.model = full_model.get_encoder()
        self.model.eval()

        if weight_path is not None:
            self.load_weights(weight_path)

        self.preprocess = nn.Conv1d(
            self.model.config.d_model,
            self.out_channels,
            1
        )
        
        # resblock
        self.resblock = ResNetBlock(
            channels=self.out_channels,
            kernel_size=3,
        )

        self.projection = nn.Conv1d(
            self.out_channels,
            self.out_channels*2,
            1
        )


    def load_weights(self, weight_path: str) -> None:
        """
        指定されたパスからモデルの重み(state_dict)をロードします。

        Args:
            weight_path (str): 重みファイル(.pth または .bin)へのパス
        """
        if str(weight_path).endswith(".safetensors"):
            state_dict = load_file(weight_path, device=self.device)
        else:
            state_dict = torch.load(weight_path, map_location=self.device)

        self.model.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        input_features: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Whisperエンコーダーの順伝播処理を行います。
        Hugging Faceのモデルを直接呼び出すことで、内部の各レイヤー処理をまとめて実行します。
        入力は30秒(パディング済み)であることを前提とします。

        Args:
            input_features (torch.Tensor): 入力音声のメルスペクトログラム特徴量。
            spec (torch.Tensor): specの長さに合わせるための参照テンソル。
            spec_lengths (torch.Tensor): specの長さ。

        Returns:
            torch.Tensor: エンコーダーの最終層の出力。
        """
        # マスクの作成（specの長さに基づく）
        target_length = spec.size(2)
        text_mask = self.generate_mask(
            spec_lengths,
            target_length,
            input_features.dtype
        )

        with torch.no_grad():
            # Whisperが30秒分の長さを期待するためパディングする
            input_features = self.input_features_padding(
                input_features
            )
            # Whisperエンコーダーの順伝播
            output = self.model(input_features).last_hidden_state

            # (Batch, Time, Dim) -> (Batch, Dim, Time)
            output = output.transpose(1, 2)

            # specの長さに合わせてinterpolate
            output = F.interpolate(
                output,
                size=target_length,
                mode='linear',
                align_corners=False
            )

        # 線形変換
        output = self.preprocess(output * text_mask)

        # Resblock
        output = self.resblock(output, text_mask)

        # maskを適用
        output = self.projection(output * text_mask)

        m, logs = output.split(self.out_channels, dim=1)

        return output, m, logs, text_mask

    def generate_mask(
        self,
        lengths: torch.Tensor,
        max_length: int,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        長さテンソルからマスクを生成します。

        Args:
            lengths (torch.Tensor): 長さのテンソル
            max_length (int): マスクの最大長
            dtype (torch.dtype): 出力マスクのデータ型

        Returns:
            torch.Tensor: 生成されたマスク [Batch, 1, Time]
        """
        progression = torch.arange(
            max_length,
            dtype=lengths.dtype,
            device=lengths.device
        )
        mask = (progression.unsqueeze(0) < lengths.unsqueeze(1))
        return torch.unsqueeze(mask, 1).to(dtype)

    def input_features_padding(self, input_features):
        """
        入力特徴量を30秒(3000フレーム)にパディングし、アテンションマスクを生成します。

        Args:
            input_features (torch.Tensor): [Batch, 80, Time]

        Returns:
            tuple: (padded_input_features, effective_time_steps)
        """
        batch, channels, time_steps = input_features.shape
        target_length = 3000  # 30秒分の長さ

        if time_steps < target_length:
            pad_amount = target_length - time_steps
            # パディング
            padded_input_features = F.pad(
                input_features,
                (0, pad_amount),
                value=0.0
            )
        else:
            # すでに3000フレーム以上の場合は、安全のために切り詰める
            padded_input_features = input_features[:, :, :target_length]

        return padded_input_features


if __name__ == "__main__":
    whisper_cocodec = WhisperEncoder(
        out_channels=192,
        weight_path=(
            "module/"
            "whisper-small-ja_voice_pseudo_whisper/"
            "checkpoint-8200/"
            "model.safetensors"
        )
    )

    # Test input_features_padding
    print("\n--- Test input_features_padding ---")
    features = torch.randn(3, 80, 6).to(whisper_cocodec.device)  # 1 second
    spec = torch.randn(3, 80, 6).to(whisper_cocodec.device)
    spec_lengths = torch.tensor([3, 4, 1]).to(whisper_cocodec.device)
    padded = whisper_cocodec.input_features_padding(features)
    print(f"Original shape: {features.shape}")
    print(f"Padded shape: {padded.shape}")

    output, m, logs, text_mask = whisper_cocodec(features, spec, spec_lengths)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output[:, 0, :]}")
    print(f"Output: {output[:, 1, :]}")
    print(f"Output: {output[:, 2, :]}")
    print(f"Output: {output[:, 3, :]}")
    print(f"Output: {output[:, 4, :]}")
