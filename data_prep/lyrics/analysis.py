import json


JSON_FILE = "data/lyrics_chunks.json"

BUCKETS = [
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (15, float("inf"))
]


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


stats = {}

for low, high in BUCKETS:

    bucket_name = (
        f"{low}-{high}"
        if high != float("inf")
        else "15+"
    )

    stats[bucket_name] = {
        "chunks": 0,
        "seconds": 0.0
    }


all_durations = []

negative_chunks = []
over_15_chunks = []
duration_mismatches = []

total_chunks = 0
total_seconds = 0.0


# =====================================================
# ANALYSIS
# =====================================================

for song_id, chunks in data.items():

    for chunk in chunks:

        start = chunk["start"]
        end = chunk["end"]
        duration = chunk["duration"]

        actual_duration = round(end - start, 3)

        all_durations.append(duration)

        total_chunks += 1
        total_seconds += duration

        # negative duration
        if duration < 0:
            negative_chunks.append(
                (song_id, chunk)
            )

        # >15 sec
        if duration > 15:
            over_15_chunks.append(
                (song_id, chunk)
            )

        # mismatch
        if abs(duration - actual_duration) > 0.01:
            duration_mismatches.append(
                (
                    song_id,
                    duration,
                    actual_duration,
                    chunk
                )
            )

        # bucketing
        for low, high in BUCKETS:

            if low <= duration < high:

                bucket_name = (
                    f"{low}-{high}"
                    if high != float("inf")
                    else "15+"
                )

                stats[bucket_name]["chunks"] += 1
                stats[bucket_name]["seconds"] += duration

                break


# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 100)
print("GLOBAL STATISTICS")
print("=" * 100)

print(f"Total chunks            : {total_chunks:,}")
print(f"Total hours             : {total_seconds / 3600:.2f}")
print(f"Min duration            : {min(all_durations):.3f}")
print(f"Max duration            : {max(all_durations):.3f}")
print(f"Negative durations      : {len(negative_chunks):,}")
print(f"Chunks >15 sec          : {len(over_15_chunks):,}")
print(f"Duration mismatches     : {len(duration_mismatches):,}")

print("\n")


# =====================================================
# BUCKET TABLE
# =====================================================

print("=" * 100)
print("DURATION DISTRIBUTION")
print("=" * 100)

print(
    f"{'Range':<12}"
    f"{'Chunks':>12}"
    f"{'%Chunks':>12}"
    f"{'Hours':>12}"
    f"{'%Hours':>12}"
)

print("-" * 100)

for bucket, values in stats.items():

    chunk_count = values["chunks"]

    hours = values["seconds"] / 3600

    pct_chunks = (
        100 * chunk_count / total_chunks
        if total_chunks else 0
    )

    pct_hours = (
        100 * values["seconds"] / total_seconds
        if total_seconds else 0
    )

    print(
        f"{bucket:<12}"
        f"{chunk_count:>12,}"
        f"{pct_chunks:>11.2f}%"
        f"{hours:>12.2f}"
        f"{pct_hours:>11.2f}%"
    )

print("-" * 100)


# =====================================================
# BAD CHUNKS
# =====================================================

if negative_chunks:

    print("\n")
    print("=" * 100)
    print("NEGATIVE DURATION EXAMPLES")
    print("=" * 100)

    for song_id, chunk in negative_chunks[:10]:

        print(f"\nSong: {song_id}")
        print(chunk)


if over_15_chunks:

    print("\n")
    print("=" * 100)
    print(">15 SECOND CHUNK EXAMPLES")
    print("=" * 100)

    for song_id, chunk in over_15_chunks[:10]:

        print(f"\nSong: {song_id}")
        print(
            f"Duration: {chunk['duration']:.3f}"
        )
        print(chunk)


if duration_mismatches:

    print("\n")
    print("=" * 100)
    print("DURATION MISMATCH EXAMPLES")
    print("=" * 100)

    for (
        song_id,
        stored,
        actual,
        chunk
    ) in duration_mismatches[:10]:
        pass