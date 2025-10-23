import os
import subprocess

video_dir = "videos"
audio_dir = "audios"

# Create output folder if it doesn’t exist
os.makedirs(audio_dir, exist_ok=True)

# Pick only .webm files
files = [f for f in os.listdir(video_dir) if f.lower().endswith(".webm")]

for i, file in enumerate(sorted(files), start=1):
    tutorial_number = f"{i:02d}"  # numbering 01, 02, 03 ...
    file_name = os.path.splitext(file)[0]  # remove extension

    print(f"Converting: {tutorial_number} - {file_name}")

    input_path = os.path.join(video_dir, file)
    mp3_output_path = os.path.join(audio_dir, f"{tutorial_number}_{file_name}.mp3")

    # Convert videos -> mp3
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-vn", "-ab", "192k", "-ar", "44100", "-y", mp3_output_path
    ], check=True)

print("✅ Done! All .webm files converted to .mp3")
