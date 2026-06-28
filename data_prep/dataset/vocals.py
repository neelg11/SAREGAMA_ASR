import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/downloads/audio")
OUTPUT_DIR = Path("data/downloads/vocals")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mp3_files = sorted(INPUT_DIR.glob("*.mp3"))

if not mp3_files:
    print(f"No MP3 files found in: {INPUT_DIR}")
    exit()

print(f"Found {len(mp3_files)} MP3 files")

for mp3 in tqdm(mp3_files, desc="Extracting vocals", unit="file"):
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "demucs",
                "--two-stems",
                "vocals",
                str(mp3),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        vocals_file = (
            Path("separated")
            / "htdemucs"
            / mp3.stem
            / "vocals.wav"
        )

        if vocals_file.exists():
            shutil.move(
                str(vocals_file),
                str(OUTPUT_DIR / f"{mp3.stem}.wav")
            )

    except Exception as e:
        print(f"\nFailed: {mp3.name} -> {e}")

# Clean temporary Demucs output
shutil.rmtree("separated", ignore_errors=True)

print(f"\nDone. Vocals saved to: {OUTPUT_DIR}")