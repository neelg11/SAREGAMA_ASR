import json
import re
from pathlib import Path


# =====================================================
# CONFIG
# =====================================================

MAX_DURATION = 15.0

MAX_REASONABLE_TIMESTAMP = 10000  # ~2.7 hours


# =====================================================
# TIME PARSER
# =====================================================

def srt_time_to_seconds(time_str):

    time_str = time_str.strip()

    if "," in time_str:
        time_part, ms = time_str.split(",", 1)

    elif "." in time_str:
        time_part, ms = time_str.split(".", 1)

    else:
        time_part = time_str
        ms = "0"

    parts = time_part.split(":")

    if len(parts) == 3:

        h, m, s = parts

    elif len(parts) == 2:

        h = 0
        m, s = parts

    else:

        raise ValueError(
            f"Invalid timestamp: {time_str}"
        )

    return (
        int(h) * 3600
        + int(m) * 60
        + int(s)
        + int(ms) / 1000.0
    )


# =====================================================
# MUSIC / CHORUS DETECTOR
# =====================================================

def is_music(text):

    t = text.strip().lower()

    patterns = [
        "~music~",
        "[music]",
        "music",
        "~chorus~",
        "[chorus]",
        "chorus",
        "♪",
    ]

    return any(p in t for p in patterns)


# =====================================================
# PARSE SRT
# =====================================================

def parse_srt(srt_file):

    with open(
        srt_file,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        content = f.read()

    blocks = re.split(
        r"\n\s*\n",
        content,
    )

    entries = []

    bad_timestamp_count = 0

    for block in blocks:

        lines = [
            x.strip()
            for x in block.splitlines()
            if x.strip()
        ]

        if len(lines) < 2:
            continue

        timestamp_line = None

        for line in lines:

            if "-->" in line:
                timestamp_line = line
                break

        if timestamp_line is None:
            continue

        try:

            start_str, end_str = [
                x.strip()
                for x in timestamp_line.split("-->")
            ]

            start = srt_time_to_seconds(start_str)
            end = srt_time_to_seconds(end_str)

            # ---------------------------------
            # CORRUPTION CHECKS
            # ---------------------------------

            if (
                start > MAX_REASONABLE_TIMESTAMP
                or end > MAX_REASONABLE_TIMESTAMP
            ):
                bad_timestamp_count += 1
                continue

            if end <= start:
                bad_timestamp_count += 1
                continue

        except Exception:

            bad_timestamp_count += 1
            continue

        text_lines = []

        timestamp_seen = False

        for line in lines:

            if "-->" in line:
                timestamp_seen = True
                continue

            if timestamp_seen:
                text_lines.append(line)

        text = " ".join(text_lines)
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            continue

        entries.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    return entries, bad_timestamp_count


# =====================================================
# CHUNK CREATION
# =====================================================

def create_chunks(
    entries,
    max_duration=15.0,
):

    chunks = []

    current = None

    for entry in entries:

        # ---------------------------------
        # Music / Chorus boundary
        # ---------------------------------

        if is_music(entry["text"]):

            if current:

                duration = round(
                    current["end"]
                    - current["start"],
                    3,
                )

                if duration > 0:

                    current["duration"] = duration

                    chunks.append(current)

                current = None

            continue

        # ---------------------------------

        if current is None:

            current = {
                "start": entry["start"],
                "end": entry["end"],
                "text": entry["text"],
            }

            continue

        proposed_duration = (
            entry["end"]
            - current["start"]
        )

        if proposed_duration <= max_duration:

            current["end"] = entry["end"]

            current["text"] += (
                " " + entry["text"]
            )

        else:

            duration = round(
                current["end"]
                - current["start"],
                3,
            )

            if duration > 0:

                current["duration"] = duration

                chunks.append(current)

            current = {
                "start": entry["start"],
                "end": entry["end"],
                "text": entry["text"],
            }

    if current:

        duration = round(
            current["end"]
            - current["start"],
            3,
        )

        if duration > 0:

            current["duration"] = duration

            chunks.append(current)

    return chunks


# =====================================================
# MAIN PROCESSOR
# =====================================================

def process_directory(
    srt_dir,
    output_json,
):

    result = {}

    total_bad_timestamps = 0
    total_chunks = 0

    srt_files = sorted(
        Path(srt_dir).glob("*.srt")
    )

    print(
        f"\nFound {len(srt_files)} SRT files\n"
    )

    for srt_file in srt_files:

        entries, bad_count = parse_srt(
            srt_file
        )

        total_bad_timestamps += bad_count

        chunks = create_chunks(
            entries,
            MAX_DURATION,
        )

        result[srt_file.stem] = chunks

        total_chunks += len(chunks)

        print(
            f"{srt_file.stem}: "
            f"{len(entries)} subtitles -> "
            f"{len(chunks)} chunks "
            f"(bad_ts={bad_count})"
        )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)

    print(
        f"Songs               : {len(result)}"
    )

    print(
        f"Total chunks        : {total_chunks}"
    )

    print(
        f"Bad timestamps skip : {total_bad_timestamps}"
    )

    print(
        f"Saved -> {output_json}"
    )

    print("=" * 80)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    process_directory(
        srt_dir="data/downloads/srt",
        output_json="data/lyrics_chunks.json",
    )