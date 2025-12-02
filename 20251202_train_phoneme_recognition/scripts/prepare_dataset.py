import os
import argparse
import random
from pathlib import Path


def prepare_dataset(test_rate=0.2, seed=1234):
    random.seed(seed)

    # ソースパスを定義
    project_root = Path(__file__).resolve().parent.parent
    master_root = project_root.parent

    sources = {
        "Nonpara30": master_root / "dataset/preprocessed/jvs_ver1/nonpara30",
        "Nonpara30w": (
            master_root / "dataset/preprocessed/jvs_ver1/nonpara30w_ver2"
        ),
        "ThroatMic": master_root / "dataset/preprocessed/throat_microphone"
    }

    # Create dataset directory
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    datasets_to_process = list(sources.keys())

    train_files = []
    test_files = []

    # Create symlinks and collect files
    for name in datasets_to_process:
        src_path = sources[name]
        if not src_path.exists():
            print(f"Warning: Source path {src_path} does not exist. Skipping.")
            continue

        link_path = dataset_dir / name
        if not link_path.exists():
            try:
                os.symlink(src_path, link_path, target_is_directory=True)
            except OSError:
                print(
                    f"Warning: Could not create symlink for {name}."
                    f" Files will be stored with absolute paths."
                )

        # Collect files for the current dataset
        current_dataset_files = []
        print(f"Scanning {src_path}...")
        for msp_file in src_path.rglob("*ppgmat.npy"):
            if link_path.exists():
                rel_path = f"{name}/{msp_file.relative_to(src_path)}"
                base_path = rel_path.replace("_ppgmat.npy", "")
                current_dataset_files.append(base_path)
            else:
                base_path = str(msp_file).replace("_ppgmat.npy", "")
                current_dataset_files.append(base_path)

        if not current_dataset_files:
            print(f"No files found for dataset {name}.")
            print("Skipping split for this dataset.")
            continue

        # Shuffle and split for the current dataset
        random.shuffle(current_dataset_files)
        split_idx = int(len(current_dataset_files) * test_rate)

        test_files.extend(current_dataset_files[:split_idx])
        train_files.extend(current_dataset_files[split_idx:])

    if not train_files and not test_files:
        print("No files found across all datasets!")
        return

    # Sort the final aggregated lists
    train_files = sorted(train_files)
    test_files = sorted(test_files)

    # Write to train.txt / test.txt
    with open(dataset_dir / "train.txt", "w", encoding="utf-8") as f:
        for item in train_files:
            f.write(f"{item}\n")

    with open(dataset_dir / "test.txt", "w", encoding="utf-8") as f:
        for item in test_files:
            f.write(f"{item}\n")

    print(f"Created train.txt ({len(train_files)} samples) and test.")
    print(f"Created test.txt ({len(test_files)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    prepare_dataset(args.test_rate, args.seed)
