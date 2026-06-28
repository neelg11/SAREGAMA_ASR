import json
from pathlib import Path

JSON_FILE = "data/lyrics_chunks.json"
HINDI_DIR = Path("data/hindi_lyrics_NO_TS")

OUTPUT_JSON = "data/chunked_lyrics_with_hindi.json"
REVIEW_JSON = "data/alignment_review.json"


def count_words(text):
    return len(text.split())


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

review_data = {}

for song_id, chunks in data.items():
    hindi_path = HINDI_DIR / f"{song_id}.txt"

    if not hindi_path.exists():
        print(f"[MISSING] {song_id}")
        continue

    with open(hindi_path, "r", encoding="utf-8") as f:
        hindi_lines = [line.strip() for line in f if line.strip()]

    hindi_idx = 0

    for chunk_idx, chunk in enumerate(chunks):
        target_words = count_words(chunk["text"])

        collected = []
        current_words = 0

        while hindi_idx < len(hindi_lines):
            next_line = hindi_lines[hindi_idx]
            next_words = count_words(next_line)

            before_diff = abs(target_words - current_words)
            after_diff = abs(target_words - (current_words + next_words))

            if after_diff <= before_diff:
                collected.append(next_line)
                current_words += next_words
                hindi_idx += 1
            else:
                break

        chunk["hindi_text"] = " ".join(collected)
        chunk["roman_word_count"] = target_words
        chunk["hindi_word_count"] = current_words
        chunk["word_diff"] = current_words - target_words

        if abs(chunk["word_diff"]) > 3:
            if song_id not in review_data:
                review_data[song_id] = []

            review_data[song_id].append({
                "chunk_index": chunk_idx,
                "start": chunk["start"],
                "end": chunk["end"],
                "roman_word_count": target_words,
                "hindi_word_count": current_words,
                "word_diff": current_words - target_words,
                "roman_text": chunk["text"],
                "hindi_text": chunk["hindi_text"]
            })

    remaining = len(hindi_lines) - hindi_idx

    if remaining > 0:
        if song_id not in review_data:
            review_data[song_id] = []

        review_data[song_id].append({
            "issue": "unused_hindi_lines",
            "remaining_lines": remaining
        })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

with open(REVIEW_JSON, "w", encoding="utf-8") as f:
    json.dump(review_data, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUTPUT_JSON}")
print(f"Saved review file: {REVIEW_JSON}")
print(f"Songs needing review: {len(review_data)}")