import whisper
import json
import os
import re

# 🔥 Use better model for accuracy (change to "base" if slow)
model = whisper.load_model("base")

# Ensure output folder exists
os.makedirs("jsons", exist_ok=True)

audios = os.listdir("audios")

# 🧹 Clean text function
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text

for audio in audios:
    if "_" in audio and audio.endswith(".mp3"):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]

        print("Processing:", number, title)

        # ✅ FIXED transcription (auto language + no forced Hindi)
        result = model.transcribe(
            audio=f"audios/{audio}",
            task="transcribe"
        )

        chunks = []

        for segment in result["segments"]:
            text = clean_text(segment["text"])

            # ❌ skip useless/noisy small chunks
            if len(text) < 5:
                continue

            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": text
            })

        chunks_with_metadata = {
            "chunks": chunks,
            "text": clean_text(result["text"])
        }

        # Save JSON
        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, indent=4, ensure_ascii=False)

print("✅ All audio files converted to clean JSON")