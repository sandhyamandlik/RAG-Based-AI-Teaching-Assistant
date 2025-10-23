import whisper
import json
import ffmpeg

# Input and temporary trimmed file
input_file = "audios/01_Basic-Structure-of-an-HTML-Website-Sigma-Web-Development.mp3"
trimmed_file = "audios/temp.mp3"

# Trim first 10 seconds
ffmpeg.input(input_file, t=10).output(trimmed_file).run(overwrite_output=True)

# Load model
model = whisper.load_model("base")

# Transcribe + translate Hindi → English
result = model.transcribe(
    trimmed_file,
    language="hi",
    task="translate",
    word_timestamps=False
)

chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks,f)