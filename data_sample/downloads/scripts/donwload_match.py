import os

# =========================
# PATHS
# =========================

AUDIO_FOLDER = "downloads/audio"
SRT_FOLDER = "downloads/srt"

# =========================
# GET AUDIO IDS
# =========================

audio_ids = set()

for file in os.listdir(AUDIO_FOLDER):

    if file.endswith(".mp3"):

        # INH100043820.mp3 -> INH100043820
        base = os.path.splitext(file)[0]

        audio_ids.add(base)

# =========================
# GET SRT IDS
# =========================

srt_ids = set()

for file in os.listdir(SRT_FOLDER):

    if file.endswith("_E.srt"):

        # INH100043820_E.srt -> INH100043820
        base = file.replace("_E.srt", "")

        srt_ids.add(base)

# =========================
# MATCHING
# =========================

matched = audio_ids & srt_ids
missing = audio_ids - srt_ids

print(f"Total audio files: {len(audio_ids)}")
print(f"Total matching lyrics: {len(matched)}")
print(f"Audio without lyrics: {len(missing)}")

# optional: print missing files
print("\nMissing lyrics for:")

for x in sorted(missing):
    print(x)