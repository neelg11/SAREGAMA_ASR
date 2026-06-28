import json
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# =========================
# CONFIG
# =========================

JSON_FILE = Path("data/training_data_final/clean_dataset.json")
VOCAL_DIR = Path("data/downloads/vocals")
OUTPUT_DIR = Path("data/training_data_final/chunks")

# =========================
# LOAD JSON
# =========================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CHUNK SONGS
# =========================

for song_id, segments in tqdm(data.items(), desc="Processing songs"):

    wav_id = song_id.replace("_E", "")
    wav_path = VOCAL_DIR / f"{wav_id}.wav"

    if not wav_path.exists():
        print(f"Missing: {wav_path}")
        continue

    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        print(f"Failed loading {wav_path}: {e}")
        continue

    song_out_dir = OUTPUT_DIR / song_id
    song_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(segments, start=1):

        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        if end_ms <= start_ms:
            continue

        chunk = audio[start_ms:end_ms]

        out_file = song_out_dir / f"{idx:05d}.wav"

        chunk.export(
            out_file,
            format="wav"
        )

print("\nDone.")