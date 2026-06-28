import os
import paramiko

# =========================
# CONFIG
# =========================

HOST = "14.140.235.226"
PORT = 22022
USERNAME = "sftp-preeti"
PASSWORD = "W!nG5t05LY"

# folder containing mp3 files
INPUT_MP3_FOLDER = "downloads/audio"

# folder where downloaded srt files will be saved
OUTPUT_SRT_FOLDER = "downloads/srt"

# =========================
# HELPER
# =========================

def exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except:
        return False

def get_candidate_paths(isrc):
    return [
        f"/data/Lyrics/{isrc}_E.srt",
    ]

# create output folder if not exists
os.makedirs(OUTPUT_SRT_FOLDER, exist_ok=True)

transport = None
sftp = None

try:
    print("Connecting to SFTP server...")

    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USERNAME, password=PASSWORD)

    sftp = paramiko.SFTPClient.from_transport(transport)

    print("✅ Connected\n")

    mp3_files = [f for f in os.listdir(INPUT_MP3_FOLDER) if f.endswith(".mp3")]

    for mp3_file in mp3_files:

        # remove .mp3 extension
        isrc = os.path.splitext(mp3_file)[0]

        print(f"\nSearching SRT for: {isrc}")

        found = False

        candidate_paths = get_candidate_paths(isrc)

        for remote_path in candidate_paths:

            if exists(sftp, remote_path):

                filename = os.path.basename(remote_path)

                local_save_path = os.path.join(
                    OUTPUT_SRT_FOLDER,
                    filename
                )

                print(f"✅ Found: {remote_path}")
                print(f"⬇ Downloading to: {local_save_path}")

                sftp.get(remote_path, local_save_path)

                found = True
                break

        if not found:
            print("❌ No SRT found")

finally:

    if sftp:
        sftp.close()

    if transport:
        transport.close()

    print("\nConnection closed.")