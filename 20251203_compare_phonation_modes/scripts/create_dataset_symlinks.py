import os
from pathlib import Path


def create_symlinks():
    # Define the project root relative to this script
    # Script is in 20251203_compare_phonation_modes/scripts/
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"

    # Ensure dataset directory exists
    if not dataset_dir.exists():
        print(f"Creating directory: {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=True)

    # Define source paths (relative to project_root)
    # Note: These paths are relative to the project root.
    # If the dataset structure changes, these might need to be adjusted.
    sources = {
        "nonpara30w_ver2": (
            "dataset/preprocessed/jvs_ver1/nonpara30w_ver2"),
        "whisper10": (
            "dataset/preprocessed/jvs_ver1/whisper10"),
        "whisper_converted_v2": (
            "dataset/pseudo_whisper_vits/whisper_converted_v2")
    }

    # Construct full source paths relative to project_root
    sources = {k: project_root / v for k, v in sources.items()}

    for link_name, source_path_str in sources.items():
        source_path = Path(source_path_str)
        link_path = dataset_dir / link_name

        if not source_path.exists():
            print(f"Warning: Source path does not exist: {source_path}")
            continue

        if link_path.exists():
            print(f"Link already exists: {link_path}")
            # Check if it points to the correct location
            if link_path.is_symlink():
                target = link_path.resolve()
                if target == source_path.resolve():
                    print("  -> Points to correct location.")
                else:
                    print(f"  -> Points to {target}, expected {source_path}")
            else:
                print("  -> Is not a symlink. Skipping.")
            continue

        print(f"Creating symlink: {link_path} -> {source_path}")
        try:
            # On Windows, target_is_directory=True is required
            # for directory symlinks
            os.symlink(source_path, link_path, target_is_directory=True)
            print("  -> Success")
        except OSError as e:
            print(f"  -> Failed: {e}")
            print("  Note: On Windows, you may need Developer Mode enabled"
                  "or run as Administrator.")


if __name__ == "__main__":
    create_symlinks()
