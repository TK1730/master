import argparse
import os
import sys
import torch
import librosa
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    GenerationConfig
)
from glob import glob


def get_latest_checkpoint(output_dir):
    checkpoints = glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Extract numbers and sort
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


def load_audio(path):
    # Whisper expects 16kHz audio
    audio, _ = librosa.load(path, sr=16000)
    return audio


def transcribe(model, processor, generation_config, audio_path, device):
    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)

    # Feature extraction
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    input_features = input_features.to(device)

    # Generate token ids
    # forced_decoder_ids are usually saved in the config,
    # but we can set language/task explicit to be safe
    # The model config should already have the fine-tuned settings
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            generation_config=generation_config,
            language="Japanese",
            task="transcribe"
        )

    # Decode token ids to text
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Inference with trained Whisper model"
    )
    parser.add_argument(
        "--audio_path", nargs="?", help="Path to audio file"
    )
    parser.add_argument(
        "--model_dir", default="./output/whisper-small-ja-wer",
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        help="Specific checkpoint path"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Resolve model path
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = get_latest_checkpoint(args.model_dir)
        if not model_path:
            # Fallback to model_dir if it is the model itself
            if os.path.exists(os.path.join(args.model_dir, "config.json")):
                model_path = args.model_dir
            else:
                print(f"Error: No checkpoints found in {args.model_dir}")
                sys.exit(1)

    print(f"Loading model from: {model_path}")
    print(f"Device: {args.device}")

    device = torch.device(args.device)
    try:
        # Load processor and model
        # Note: Often the processor is not saved in the checkpoint dir
        # unless specifically saved.
        # If not found, we might need to load from the base model,
        # but let's try the checkpoint first.
        try:
            processor = WhisperProcessor.from_pretrained(model_path)
        except (OSError, TypeError):
            print(
                "Warning: Processor not found in checkpoint, "
                "loading from 'openai/whisper-small'"
            )
            processor = WhisperProcessor.from_pretrained(
                "openai/whisper-small", language="Japanese", task="transcribe"
            )

        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        generation_config = GenerationConfig.from_pretrained(model_path)
        model.to(device)
        model.eval()

    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    if args.audio_path:
        audio_paths = [args.audio_path]
    else:
        # Interactive mode or list available files?
        # Let's try to grab a file from dataset if None provided?
        # For now just ask user
        print("Please provide an audio path.")
        # Try to find a wav file in dataset/audio or similar if it exists?
        # Let's just exit
        sys.exit(1)

    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue

        try:
            result = transcribe(
                model, processor, generation_config, audio_path, device
            )
            print("-" * 30)
            print(f"File: {audio_path}")
            print(f"Transcription: {result}")
            print("-" * 30)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")


if __name__ == "__main__":
    main()
