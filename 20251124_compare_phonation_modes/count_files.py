from pathlib import Path

# Count files in whisper10
whisper10_files = list(Path("dataset/whisper10").rglob("*.wav"))
print(f"Total files in whisper10: {len(whisper10_files)}")

# Count files in whisper10_topline
topline_files = list(Path("dataset/whisper10_topline").rglob("*.wav"))
print(f"Processed files in whisper10_topline: {len(topline_files)}")

print(f"Remaining files to process: {len(whisper10_files) - len(topline_files)}")
