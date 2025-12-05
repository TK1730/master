import os
from pathlib import Path


def create_symlinks():
    """
    Create symbolic links for the datasets required for the experiment.
    
    This script creates symlinks in the data/ directory pointing to:
    - dataset/preprocessed/jvs_ver1/nonpara30 (target: voiced speech)
    - dataset/whisper2voice/whisper_converted_v2 (VITS pseudo-whisper conversion)
    - dataset/whisper2voice/whisper2voice (whisper to voice conversion)
    """
    # Define the project root relative to this script
    # Script is in 20251204_analyze_pseudo_whisper_using_vits_whisper2voice/scripts/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    master_root = project_root.parent
    data_dir = project_root / "data"

    # Ensure data directory exists
    if not data_dir.exists():
        print(f"Creating directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    # Define source paths (relative to master_root)
    sources = {
        "target": master_root / "dataset/preprocessed/jvs_ver1/nonpara30",
        "vits_gen": master_root / "dataset/whisper2voice/whisper_converted_v2",
        "whisper_conv": master_root / "dataset/whisper2voice/whisper2voice"
    }

    for link_name, source_path in sources.items():
        link_path = data_dir / link_name

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
            print("  Note: On Windows, you may need Developer Mode enabled "
                  "or run as Administrator.")


if __name__ == "__main__":
    create_symlinks()
