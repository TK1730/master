# encoding: utf-8

import math
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Components for constructing the training model
from .model_component import monotonic_align
from .model_component.decoder import Decoder
from .model_component.flow import Flow
from .model_component.posterior_encoder import PosteriorEncoder
from .model_component.stochastic_duration_predictor import (
    StochasticDurationPredictor
)
from .model_component.whisper_encoder import WhisperEncoder


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, d, t = x.size()
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    padding_shape = [[0, 0], [1, 0], [0, 0]]
    padding = [item for sublist in padding_shape[::-1] for item in sublist]

    path = path - F.pad(path, padding)[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


class VitsGenerator(nn.Module):
    def __init__(self, n_phoneme: int, n_speakers: int):
        super().__init__()
        self.n_phoneme = n_phoneme  # Input phoneme types count
        self.phoneme_embedding_dim = 192  # Dimension of phoneme embedding vector
        self.spec_channels = 513  # Dimension of linear spectrogram frequency axis
        self.z_channels = 192  # Number of z channels output from PosteriorEncoder
        self.text_encoders_dropout_during_train = 0.1  # Training dropout rate for text_encoder
        self.segment_size = 32  # Segment size for decoding z during generation
        self.n_speakers = n_speakers  # Number of speakers
        self.speaker_id_embedding_dim = 256  # Dimension of speaker ID embedding vector

        self.whisper_encoder = WhisperEncoder(
            out_channels=self.phoneme_embedding_dim,
            model_name="openai/whisper-small",
            device="cuda",
            weight_path="module/whisper-small-ja_voice_pseudo_whisper/checkpoint-8200/model.safetensors"
        )

        # Speaker embedding network
        self.speaker_embedding = nn.Embedding(
            num_embeddings=self.n_speakers,  # Number of speakers
            embedding_dim=self.speaker_id_embedding_dim  # Dimension of speaker ID embedding
        )

        # Posterior Encoder: Encodes linear spectrogram and embedded speaker ID to z
        self.posterior_encoder = PosteriorEncoder(
            speaker_id_embedding_dim=self.speaker_id_embedding_dim,
            in_spec_channels=self.spec_channels,
            out_z_channels=self.z_channels,
            phoneme_embedding_dim=self.phoneme_embedding_dim,
        )

        # Decoder: Generates audio from z and embedded speaker ID
        self.decoder = Decoder(
            speaker_id_embedding_dim=self.speaker_id_embedding_dim,
            in_z_channel=self.z_channels
        )

        # Flow: Reversible network for MAS z_p (and speaker conversion)
        self.flow = Flow(
            speaker_id_embedding_dim=self.speaker_id_embedding_dim,
            in_z_channels=self.z_channels,
            phoneme_embedding_dim=self.phoneme_embedding_dim,
        )

    def forward(
        self,
        text_padded: torch.Tensor,
        text_lengths: torch.Tensor,
        spec_padded: torch.Tensor,
        spec_lengths: torch.Tensor,
        speaker_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[Any, ...]]:
        # Pass text (phonemes) through TextEncoder
        text_encoded, m_p, logs_p, text_mask = self.whisper_encoder(
            text_padded,
            spec_padded,
            spec_lengths
        )

        # Embed speaker ID
        speaker_id_embedded = self.speaker_embedding(speaker_id).unsqueeze(-1)

        # Posterior Encoder
        z, m_q, logs_q, spec_mask = self.posterior_encoder(
            spec_padded, spec_lengths, speaker_id_embedded
        )

        # Flow
        z_p = self.flow(z, spec_mask, speaker_id_embedded=speaker_id_embedded)

        # Slice z
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        # Generate audio waveform
        wav_fake = self.decoder(z_slice, speaker_id_embedded=speaker_id_embedded)

        return (
            wav_fake,
            ids_slice,
            text_mask,
            spec_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q)
        )

    def text_to_speech(
        self,
        wav_padded: torch.Tensor,
        spec_padded: torch.Tensor,
        spec_lengths: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> torch.Tensor:
        text_encoded, m_p, logs_p, text_mask = self.whisper_encoder(
            wav_padded,
            spec_padded,
            spec_lengths
        )
        speaker_id_embedded = self.speaker_embedding(speaker_id).unsqueeze(-1)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z = self.flow(
            z_p, text_mask, speaker_id_embedded=speaker_id_embedded, reverse=True
        )
        wav_fake = self.decoder(
            (z * text_mask), speaker_id_embedded=speaker_id_embedded
        )
        return wav_fake

    def voice_conversion(
        self,
        spec_padded: torch.Tensor,
        spec_lengths: torch.Tensor,
        source_speaker_id: torch.Tensor,
        target_speaker_id: torch.Tensor
    ) -> torch.Tensor:
        assert self.n_speakers > 0
        emb_source = self.speaker_embedding(source_speaker_id).unsqueeze(-1)
        emb_target = self.speaker_embedding(target_speaker_id).unsqueeze(-1)
        z, m_q, logs_q, spec_mask = self.posterior_encoder(
            spec_padded, spec_lengths, speaker_id_embedded=emb_source
        )
        z_p = self.flow(z, spec_mask, speaker_id_embedded=emb_source)
        z_hat = self.flow(
            z_p, spec_mask, speaker_id_embedded=emb_target, reverse=True
        )
        wav_fake = self.decoder(z_hat * spec_mask, speaker_id_embedded=emb_target)
        return wav_fake
