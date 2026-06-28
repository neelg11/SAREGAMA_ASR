import json
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

DATA_DIR = Path("data/training_data_final")

LYRICS_JSON = DATA_DIR / "clean_dataset.json"
SPLIT_JSON = DATA_DIR / "split.json"

CHUNKS_DIR = DATA_DIR / "chunks"

MANIFEST_DIR = DATA_DIR / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

with open(LYRICS_JSON, "r", encoding="utf-8") as f:
    lyrics_data = json.load(f)

with open(SPLIT_JSON, "r", encoding="utf-8") as f:
    split_data = json.load(f)

# =====================================================
# CREATE MANIFESTS
# =====================================================

for split_name in ["train", "val", "test"]:

    songs = split_data[split_name]

    output_file = MANIFEST_DIR / f"{split_name}.jsonl"

    examples = 0
    missing_chunks = 0

    with open(output_file, "w", encoding="utf-8") as fout:

        for song_id in songs:

            if song_id not in lyrics_data:
                print(f"Lyrics missing: {song_id}")
                continue

            segments = lyrics_data[song_id]

            for idx, seg in enumerate(segments, start=1):

                chunk_path = CHUNKS_DIR / song_id / f"{idx:05d}.wav"

                if not chunk_path.exists():
                    missing_chunks += 1
                    continue

                text = seg["hindi_text"].strip()

                if not text:
                    continue

                record = {
                    "audio": str(chunk_path),
                    "text": text
                }

                fout.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                examples += 1

    print(
        f"{split_name}: {examples:,} examples | "
        f"missing chunks: {missing_chunks:,}"
    )

print("\nFinished.")