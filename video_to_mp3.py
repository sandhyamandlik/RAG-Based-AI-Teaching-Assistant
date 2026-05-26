import os
import subprocess

video_dir = "videos"
audio_dir = "audios"

os.makedirs(audio_dir, exist_ok=True)

all_files = os.listdir(video_dir)
files = [f for f in all_files if f.lower().endswith((".webm", ".mp4", ".mkv"))]

print("Files found in videos:", all_files)
print("Filtered video files:", files)

for i, file in enumerate(sorted(files), start=1):
    tutorial_number = f"{i:02d}"
    file_name = os.path.splitext(file)[0]

    print(f"Converting: {tutorial_number} - {file_name}")

    input_path = os.path.join(video_dir, file)
    mp3_output_path = os.path.join(audio_dir, f"{tutorial_number}_{file_name}.mp3")

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-vn", "-ab", "192k", "-ar", "44100", mp3_output_path
    ], check=True)

print("✅ Done!")