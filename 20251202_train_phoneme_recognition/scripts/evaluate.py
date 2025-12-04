"""
モデル評価スクリプト
dataset/test.txtのデータを使って音素認識の精度とヒートマップ描画を行います。
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.lstm_net_rev import LSTM_net  # E402
from scripts.dataset import PhonemeDataset, collate_fn  # E402


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config_params):
    model = LSTM_net(
        n_inputs=config_params["n_inputs"],
        n_outputs=config_params["n_outputs"],
        n_layers=config_params["n_layers"],
        hidden_size=config_params["hidden_size"],
        fc_size=config_params["fc_size"],
        dropout=config_params["dropout"],
        bidirectional=config_params["bidirectional"],
        l2softmax=config_params["l2softmax"],
        continuous=config_params["continuous"]
    )
    return model


def get_latest_model_path(trained_models_dir):
    # Find the latest directory in trained_models
    dirs = [d for d in trained_models_dir.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No trained models found.")
    # Sort by name (timestamp)
    dirs = sorted(dirs, key=lambda x: x.name)
    latest_dir = dirs[-1]

    # Check for model.pth or checkpoint.pth
    model_path = latest_dir / "model.pth"
    if not model_path.exists():
        model_path = latest_dir / "checkpoint.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"No model file found in {latest_dir}")

    return model_path, latest_dir


def evaluate_pattern(model, dataloader, device, pattern_name, output_dir):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for msp, ppg in dataloader:
            msp = msp.to(device)
            ppg = ppg.to(device)

            # Forward
            outputs = model(msp)  # (Batch, Time, Class)

            # Get predictions
            preds = torch.argmax(outputs, dim=2)  # (Batch, Time)
            labels = torch.argmax(ppg, dim=2)     # (Batch, Time)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(36)))

    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title(f'Confusion Matrix - {pattern_name}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_dir / f"heatmap_{pattern_name}.png")
    plt.close()

    return accuracy


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    trained_models_dir = project_root / "trained_models"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model Path first to get model_dir
    try:
        model_path, model_dir = get_latest_model_path(trained_models_dir)
        print(f"Loading model from: {model_path}")
    except FileNotFoundError as e:
        print(e)
        return

    # Load Config from model directory
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        print(f"Config not found in {model_dir}, falling back to root config.")
        config_path = project_root / "config.yaml"
        config = load_config(config_path)
        # Override for the specific model if known mismatch
        print("Overriding config with inferred "
              "parameters from error analysis:")
        print("n_outputs: 36")
        print("bidirectional: True")
        config['model_params']['n_outputs'] = 36
        config['model_params']['bidirectional'] = True
    else:
        config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    model = create_model(config['model_params'])

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    # Prepare Output Directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use the model directory name to identify which model was evaluated
    model_id = model_dir.name
    output_dir = project_root.joinpath(
        "evaluation_results",
        f"{model_id}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Dataset
    full_dataset = PhonemeDataset(
        dataset_dir / "test.txt",
        root_dir=dataset_dir
    )

    # Define Patterns
    # 1. All files
    indices_all = list(range(len(full_dataset)))

    # 2. Voiced (Nonpara30/)
    indices_voiced = [
        i for i, f in enumerate(full_dataset.files) if "Nonpara30/" in f
    ]

    # 3. Pseudo-whisper (Nonpara30w/)
    indices_pseudo = [
        i for i, f in enumerate(full_dataset.files) if "Nonpara30w/" in f
    ]

    # 4. Whisper (ThroatMic/)
    indices_whisper = [
        i for i, f in enumerate(full_dataset.files) if "ThroatMic/" in f
    ]

    patterns = {
        "all": indices_all,
        "voiced": indices_voiced,
        "pseudo_whisper": indices_pseudo,
        "whisper": indices_whisper
    }

    results = {}

    for name, indices in patterns.items():
        print(f"Evaluating pattern: {name} ({len(indices)} samples)")
        if len(indices) == 0:
            print(f"Warning: No samples found for pattern {name}")
            results[name] = 0.0
            continue

        subset = Subset(full_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=config['hyperparameters']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Avoid multiprocessing issues on Windows if any
        )

        acc = evaluate_pattern(model, loader, device, name, output_dir)
        results[name] = acc
        print(f"Accuracy for {name}: {acc:.4f}")

    # Save results to text file
    with open(output_dir / "accuracy.txt", "w") as f:
        f.write(f"Model evaluated: {model_path}\n")
        f.write("-" * 30 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
