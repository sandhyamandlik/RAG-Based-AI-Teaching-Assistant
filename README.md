#  VidMind AI — AI Teaching Assitant

> Upload videos. Transcribe them. Ask anything. Get taught by AI.

VidMind AI is a local RAG (Retrieval-Augmented Generation) system that turns your video library into a searchable, queryable knowledge base. It transcribes videos using Whisper, indexes the content with semantic embeddings via `bge-m3`, and answers questions using a locally running LLM through Ollama — entirely offline, no API keys required.

---

## Features

- **End-to-end pipeline** — Upload → Convert → Transcribe → Embed → Search, all from one UI
- **Local & private** — runs fully on your machine using Ollama; no data leaves your system
- **Semantic search** — finds relevant video chunks by meaning, not just keywords
- **AI Teaching Mode** — structured responses with explanation, real-world examples, key takeaways, and video timestamps
- **Advanced search** — filter results by specific video titles
- **Live pipeline log** — real-time per-file status during conversion, transcription, and embedding
- **Export** — download AI responses, search results (CSV), or a full report
- **FastAPI backend** — optional REST API for programmatic access

---

## 🗂 Project Structure

```
vidmind-ai/
│
├── app.py                  # Main Streamlit UI (full pipeline + search)
├── backend.py              # FastAPI REST API (optional)
├── video_to_mp3.py         # Batch video → MP3 conversion (FFmpeg)
├── mp3_to_jsons.py         # MP3 → timestamped JSON transcription (Whisper)
├── preprocess_json.py      # JSON → vector embeddings (bge-m3 via Ollama)
├── process_incoming.py     # CLI query tool (terminal-based RAG)
├── stt.py                  # Quick speech-to-text test script
├── requirements.txt        # Python dependencies
│
├── videos/                 # Place your input video files here
├── audios/                 # Auto-generated MP3 files
├── jsons/                  # Auto-generated transcript JSONs
└── embeddings.joblib       # Auto-generated vector store
```

---

## Prerequisites

### System
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in PATH
- [Ollama](https://ollama.com) installed and running

### Ollama Models
Pull these two models before running:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Start the Ollama server:

```bash
ollama serve
```

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/sandhyamandlik/RAG-Based-AI-Teaching-Assistant
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

> On some systems use `pip install -r requirements.txt --break-system-packages`

**3. Install FFmpeg**

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org and add to PATH
```

**4. Run the app**

```bash
streamlit run app.py
```

---

## Using the Interface

The UI walks you through a 5-step pipeline, tracked by a progress stepper at the top.

### Step 01 — Upload Videos
Drag and drop `.mp4`, `.mkv`, or `.webm` files directly into the browser. Files are saved to the `videos/` folder automatically.

### Step 02 — Convert to Audio
Click **Convert to MP3** to batch-convert all uploaded videos using FFmpeg. Output files land in `audios/` with numbered prefixes (`01_`, `02_`, etc.).

### Step 03 — Transcribe
Click **Transcribe** to run OpenAI Whisper on every audio file. Each file produces a timestamped JSON in `jsons/` containing per-segment text with `start` and `end` times.

> Default model: `base` (good balance of speed and accuracy). Change `whisper_model_choice` in `app.py` to `tiny`, `small`, or `medium` as needed.

### Step 04 — Build Embeddings
Click **Build Embeddings** to send all transcript chunks to Ollama's `bge-m3` model and store the resulting vectors in `embeddings.joblib`.

### Step 05 — Search & Ask
Type any question into the search box. The system:
1. Embeds your query with `bge-m3`
2. Retrieves the top-K most similar transcript chunks
3. Sends the context + question to `llama3.2`
4. Returns a structured teaching response with explanation, example, key takeaways, and timestamps

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Description |
|---|---|---|
| Top K Results | 5 | Number of transcript chunks retrieved per query |
| Min Similarity | 0.20 | Minimum cosine similarity threshold (0–1) |
| LLM | llama3.2:latest | Ollama model used for generating answers |
| Temperature | 0.70 | LLM creativity (0 = deterministic, 1 = creative) |

**Recommended settings for most use cases:**
- Temperature: `0.4` — accurate and natural
- Min Similarity: `0.25` — filters noise without missing good matches

---

## 🌐 REST API (Optional)

Start the FastAPI backend:

```bash
uvicorn backend:app --reload
```

### Endpoints

**GET `/api/status`** — Check embeddings and Ollama status

```json
{
  "embeddings": { "count": 342, "status": "ready" },
  "ollama": { "running": true, "model": "llama3.2" }
}
```

**POST `/api/query`** — Ask a question

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a CSS class?"}'
```

```json
{
  "success": true,
  "response": "A CSS class is...",
  "videos": [ { "title": "...", "start": 42.0, "end": 48.5, "similarity": 0.74 } ]
}
```

---

## CLI Usage

For quick terminal-based queries without the UI:

```bash
python process_incoming.py
```

You'll see topic suggestions and chunk previews from the indexed content, then enter your question to get an AI answer printed directly in the terminal. Responses are also saved to `response.txt`.

---

## How It Works

```
Video File
    │
    ▼
FFmpeg → MP3
    │
    ▼
Whisper → Timestamped JSON chunks
    │
    ▼
Ollama bge-m3 → Vector embeddings → embeddings.joblib
    │
    ▼
User Query → bge-m3 embedding → Cosine similarity search
    │
    ▼
Top-K chunks → llama3.2 prompt → Structured AI response
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `openai-whisper` | Audio transcription |
| `ffmpeg-python` | Video to audio conversion |
| `scikit-learn` | Cosine similarity search |
| `joblib` | Embedding store serialization |
| `numpy` | Matrix operations |
| `requests` | Ollama API communication |
| `fastapi` + `uvicorn` | REST API backend |
| `pandas` | Data handling |

---

## 🛠 Troubleshooting

**Ollama Offline** — Make sure you've run `ollama serve` in a separate terminal before launching the app.

**FFmpeg not found** — Verify with `ffmpeg -version` in your terminal. If missing, install it and ensure it's in your system PATH.

**No results returned** — Lower the Min Similarity slider (try `0.15`). This can happen on short or noisy transcript chunks.

**Slow transcription** — Switch Whisper model to `tiny` by changing `whisper_model_choice = "tiny"` in `app.py`. For longer videos, `base` or `small` gives much better accuracy.

**Embedding errors** — Ensure `bge-m3` is pulled: `ollama pull bge-m3` and that Ollama is running.

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  Built with Streamlit · Ollama · Whisper · bge-m3
</div>
