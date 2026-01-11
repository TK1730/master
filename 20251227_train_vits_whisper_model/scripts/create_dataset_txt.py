from pathlib import Path
import random


def create_dataset_files():
    # Use pathlib relative to current working directory (project root)
    dataset_dir = Path("dataset")

    nonpara_dir = dataset_dir / "nonpara30w_ver2"
    whisper_dir = dataset_dir / "whisper10"

    output_train = Path("train_dataset.txt")
    output_test = Path("test_dataset.txt")

    nonpara_data = []  # (path_str, speaker_id)
    whisper_data = []  # (path_str, speaker_id)

    print("Scanning nonpara30w_ver2...")
    # nonpara30w_ver2/jvsXXX/wav/*.wav
    if nonpara_dir.exists():
        for jvs_dir in nonpara_dir.glob("jvs*"):
            if not jvs_dir.is_dir():
                continue
            try:
                # jvs001 -> 1
                speaker_id = int(jvs_dir.name.replace("jvs", "")) + 100
            except ValueError:
                continue
            for wav_file in (jvs_dir / "wav").glob("*.wav"):
                nonpara_data.append((str(wav_file), speaker_id))
    else:
        print(f"Directory not found: {nonpara_dir}")

    print(f"Found {len(nonpara_data)} files in nonpara30w_ver2.")

    print("Scanning whisper10...")
    # whisper10/jvsXXX/wav/*.wav
    if whisper_dir.exists():
        for jvs_dir in whisper_dir.glob("jvs*"):
            if not jvs_dir.is_dir():
                continue

            try:
                speaker_id = int(jvs_dir.name.replace("jvs", ""))
            except ValueError:
                continue

            for wav_file in (jvs_dir / "wav").glob("*.wav"):
                whisper_data.append((str(wav_file), speaker_id))
    else:
        print(f"Directory not found: {whisper_dir}")

    print(f"Found {len(whisper_data)} files in whisper10.")

    # Sort data for reproducible random sampling
    nonpara_data.sort()
    whisper_data.sort()

    # Random selection for test set
    random.seed(42)

    if len(nonpara_data) < 10:
        print("Warning: nonpara30w_ver2 has fewer than 10 files!")
        test_nonpara = nonpara_data[:]
        train_nonpara = []
    else:
        test_nonpara = random.sample(nonpara_data, 10)
        test_paths_np = set(t[0] for t in test_nonpara)
        train_nonpara = [d for d in nonpara_data if d[0] not in test_paths_np]

    if len(whisper_data) < 10:
        print("Warning: whisper10 has fewer than 10 files!")
        test_whisper = whisper_data[:]
        train_whisper = []
    else:
        test_whisper = random.sample(whisper_data, 10)
        test_paths_wh = set(t[0] for t in test_whisper)
        train_whisper = [d for d in whisper_data if d[0] not in test_paths_wh]

    test_dataset = test_nonpara + test_whisper
    train_dataset = train_nonpara + train_whisper

    # Sort datasets as requested (no shuffle)
    train_dataset.sort()
    test_dataset.sort()

    print(f"Writing {len(train_dataset)} lines to {output_train}")
    # Using pathlib open
    with output_train.open("w", encoding="utf-8") as f:
        for path, sid in train_dataset:
            f.write(f"{path}|{sid}\n")

    print(f"Writing {len(test_dataset)} lines to {output_test}")
    with output_test.open("w", encoding="utf-8") as f:
        for path, sid in test_dataset:
            f.write(f"{path}|{sid}\n")

    print("Done.")


if __name__ == "__main__":
    create_dataset_files()
