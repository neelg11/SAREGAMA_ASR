import json
import os

MERGED_JSON = "data/chunked_lyrics_with_hindi.json"
REVIEW_JSON = "data/alignment_review.json"

CLEAN_JSON = "data/training_data_final/clean_dataset.json"
PROBLEMATIC_JSON = "data/problematic_dataset.json"

os.makedirs(os.path.dirname(CLEAN_JSON), exist_ok=True)

with open(MERGED_JSON, "r", encoding="utf-8") as f:
    merged = json.load(f)

with open(REVIEW_JSON, "r", encoding="utf-8") as f:
    review = json.load(f)

problematic_song_ids = set(review.keys())

clean_data = {}
problematic_data = {}

for song_id, song_data in merged.items():

    if song_id in problematic_song_ids:
        problematic_data[song_id] = song_data
    else:
        clean_data[song_id] = song_data

with open(CLEAN_JSON, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

with open(PROBLEMATIC_JSON, "w", encoding="utf-8") as f:
    json.dump(problematic_data, f, ensure_ascii=False, indent=2)

print(f"Total songs        : {len(merged)}")
print(f"Clean songs        : {len(clean_data)}")
print(f"Problematic songs  : {len(problematic_data)}")
print(f"Saved {CLEAN_JSON}")
print(f"Saved {PROBLEMATIC_JSON}")