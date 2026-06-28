import os
import time
import shutil
import paramiko

# ================= CONFIG =================

SONG_LIST_FILE = "inh.txt"

REMOTE_MP3_PATH = "/data/Audio"

SOURCE_SRT_FOLDER = "./lyrics"

LOCAL_AUDIO_FOLDER = "./downloads/audio"
LOCAL_SRT_FOLDER = "./downloads/srt"

LOG_FILE = "download_log.txt"

HOST = "14.140.235.226"
PORT = 22022
USERNAME = "sftp-preeti"
PASSWORD = "W!nG5t05LY"

RETRY_COUNT = 2
RETRY_DELAY = 2

# ==========================================

os.makedirs(LOCAL_AUDIO_FOLDER, exist_ok=True)
os.makedirs(LOCAL_SRT_FOLDER, exist_ok=True)


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def remote_exists(sftp_client, remote_path):
    try:
        sftp_client.stat(remote_path)
        return True
    except:
        return False


# clear old log
open(LOG_FILE, "w").close()

# ================= READ SONG LIST =================

with open(SONG_LIST_FILE, "r", encoding="utf-8") as f:
    content = f.read()

songs = content.split()
songs = [x.strip() for x in songs if x.strip().endswith(".mp3")]

log(f"Total songs found: {len(songs)}")

# ================= CONNECT SFTP =================

log("Connecting to SFTP...")

transport = paramiko.Transport((HOST, PORT))
transport.connect(
    username=USERNAME,
    password=PASSWORD
)

sftp = paramiko.SFTPClient.from_transport(transport)

log("SFTP connected")

# ================= DOWNLOAD =================

downloaded = 0
missing_srt = 0
missing_mp3 = 0

for mp3_name in songs:

    try:

        isrc = os.path.splitext(mp3_name)[0]

        remote_mp3 = f"{REMOTE_MP3_PATH}/{mp3_name}"

        local_mp3 = os.path.join(
            LOCAL_AUDIO_FOLDER,
            mp3_name
        )

        log(f"\n[{isrc}]")

        # ---------- CHECK REMOTE MP3 ----------

        if not remote_exists(sftp, remote_mp3):

            log("MP3 missing on server")

            missing_mp3 += 1
            continue

        # ---------- DOWNLOAD MP3 ----------

        success = False

        for attempt in range(1, RETRY_COUNT + 2):

            try:

                log(f"Downloading MP3 attempt {attempt}")

                sftp.get(remote_mp3, local_mp3)

                success = True
                break

            except Exception as e:

                log(f"Attempt failed: {e}")

                if attempt <= RETRY_COUNT:
                    time.sleep(RETRY_DELAY)

        if not success:

            log("Failed to download MP3")
            continue

        log("MP3 downloaded")

        # ---------- FIND MATCHING SRT ----------

        found_srt = False

        for srt_file in os.listdir(SOURCE_SRT_FOLDER):

            if not srt_file.lower().endswith(".srt"):
                continue

            if srt_file.startswith(isrc):

                source_srt = os.path.join(
                    SOURCE_SRT_FOLDER,
                    srt_file
                )

                local_srt = os.path.join(
                    LOCAL_SRT_FOLDER,
                    srt_file
                )

                shutil.copy2(source_srt, local_srt)

                log(f"SRT copied: {srt_file}")

                found_srt = True
                break

        if not found_srt:

            log("SRT not found")
            missing_srt += 1

        downloaded += 1

    except Exception as e:

        log(f"ERROR: {e}")

# ================= CLOSE =================

sftp.close()
transport.close()

# ================= SUMMARY =================

log("\n========== SUMMARY ==========")
log(f"Downloaded MP3s : {downloaded}")
log(f"Missing MP3s    : {missing_mp3}")
log(f"Missing SRTs    : {missing_srt}")
log(f"Audio folder    : {LOCAL_AUDIO_FOLDER}")
log(f"SRT folder      : {LOCAL_SRT_FOLDER}")

print("Done. Check download_log.txt")