import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import os
import subprocess
import json
import re
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="VidMind AI", layout="wide", page_icon="🧠")

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k, v in {
    "query": "",
    "last_results": None,
    "ai_response": "",
    "query_count": 0,
    "pipeline_log": [],
    "active_step": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  GROQ CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama3-70b-8192"

# ─────────────────────────────────────────────
#  GLOBAL STYLE  (unchanged)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background-color: #05080f !important; color: #c4cfe0; font-family: 'Inter', sans-serif; }
.stApp::before { content:''; position:fixed; top:-200px; left:-200px; width:600px; height:600px; background:radial-gradient(circle, #0d3a2e44 0%, transparent 70%); pointer-events:none; z-index:0; }
.stApp::after  { content:''; position:fixed; bottom:-200px; right:-200px; width:500px; height:500px; background:radial-gradient(circle, #1a1a4a33 0%, transparent 70%); pointer-events:none; z-index:0; }
#MainMenu, footer, header { visibility: hidden; }
.wordmark { font-family:'Syne',sans-serif; font-size:10px; font-weight:700; letter-spacing:6px; color:#2acea8; text-transform:uppercase; padding:22px 0 4px; opacity:0.6; }
.hero-title { font-family:'Syne',sans-serif; font-size:clamp(2.4rem,5vw,4.2rem); font-weight:800; line-height:1.05; background:linear-gradient(120deg,#2acea8 0%,#5ba4f5 50%,#a78bfa 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:6px; letter-spacing:-1px; }
.hero-sub { color:#2e3e55; font-size:12px; letter-spacing:4px; text-transform:uppercase; font-weight:500; margin-bottom:36px; }
.stepper-wrap { display:flex; align-items:center; margin:28px 0 36px; overflow-x:auto; padding-bottom:4px; gap:0; }
.step-node { display:flex; flex-direction:column; align-items:center; min-width:110px; }
.step-circle { width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-family:'Syne',sans-serif; font-size:14px; font-weight:700; border:1.5px solid #151f30; background:#090e18; color:#1e2d40; }
.step-circle.done   { border-color:#2acea8; color:#2acea8; background:#071912; }
.step-circle.active { border-color:#5ba4f5; color:#5ba4f5; background:#080f1e; box-shadow:0 0 20px #5ba4f530; }
.step-label { font-size:9px; letter-spacing:2px; text-transform:uppercase; color:#1e2d40; margin-top:7px; text-align:center; font-weight:500; }
.step-label.done   { color:#2acea8; }
.step-label.active { color:#5ba4f5; }
.step-line { flex:1; height:1px; background:#151f30; margin-bottom:20px; min-width:24px; }
.step-line.done { background:linear-gradient(90deg,#2acea8,transparent); }
.panel { background:#080d18; border:1px solid #111d30; border-radius:18px; padding:26px 30px; margin-bottom:20px; position:relative; overflow:hidden; }
.panel::after { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,#2acea840,#5ba4f540,#a78bfa40,transparent); }
.panel-title { font-family:'Syne',sans-serif; font-size:10px; font-weight:700; letter-spacing:4px; text-transform:uppercase; color:#2acea8; margin-bottom:18px; display:flex; align-items:center; gap:8px; }
.panel-title::before { content:''; display:inline-block; width:4px; height:4px; border-radius:50%; background:#2acea8; box-shadow:0 0 6px #2acea8; }
.chips-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:6px; }
.chip { background:#0a1020; border:1px solid #111d30; border-radius:10px; padding:14px 22px; min-width:130px; position:relative; overflow:hidden; }
.chip::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,#2acea840,transparent); }
.chip-val { font-family:'Syne',sans-serif; font-size:26px; color:#5ba4f5; font-weight:800; line-height:1.1; }
.chip-lbl { font-size:9px; letter-spacing:2.5px; text-transform:uppercase; color:#253040; margin-top:4px; font-weight:500; }
.log-box { background:#040810; border:1px solid #0e1826; border-radius:10px; padding:14px 18px; font-family:'Courier New',monospace; font-size:11.5px; color:#2acea8; max-height:180px; overflow-y:auto; line-height:2; }
.res-card { background:#070c18; border:1px solid #111d30; border-radius:14px; padding:18px 22px; margin-bottom:12px; transition:border-color .25s, box-shadow .25s; }
.res-card:hover { border-color:#2acea850; box-shadow:0 4px 24px #2acea810; }
.res-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px; gap:12px; }
.res-title { font-family:'Syne',sans-serif; font-size:12px; color:#5ba4f5; letter-spacing:0.5px; font-weight:700; line-height:1.4; }
.res-score { font-family:'Courier New',monospace; font-size:12px; color:#2acea8; background:#071912; border:1px solid #0f3028; border-radius:6px; padding:3px 10px; white-space:nowrap; flex-shrink:0; }
.res-time  { font-size:10px; color:#a78bfa; letter-spacing:1.5px; margin-bottom:10px; font-family:'Courier New',monospace; }
.res-text  { font-size:13.5px; color:#7a8fa8; line-height:1.75; }
.dot-ok   { color:#2acea8; font-size:13px; }
.dot-warn { color:#f59e0b; font-size:13px; }
.dot-bad  { color:#f43f5e; font-size:13px; }
section[data-testid="stSidebar"] { background:#040810 !important; border-right:1px solid #0e1826 !important; }
section[data-testid="stSidebar"] * { color:#5a7090 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#2acea8 !important; font-family:'Syne',sans-serif !important; letter-spacing:2px !important; }
.stButton > button { background:transparent !important; border:1px solid #1a3040 !important; color:#5ba4f5 !important; font-family:'Syne',sans-serif !important; font-size:10px !important; font-weight:700 !important; letter-spacing:2.5px !important; border-radius:6px !important; padding:7px 14px !important; transition:all .2s !important; text-transform:uppercase !important; height:34px !important; min-height:0 !important; line-height:1 !important; white-space:nowrap !important; width:100% !important; }
.stButton > button:hover { background:#5ba4f510 !important; border-color:#5ba4f5 !important; color:#5ba4f5 !important; }
.stButton > button:disabled { opacity:0.3 !important; cursor:not-allowed !important; }
div[data-testid="stFileUploader"] { background:#080d18 !important; border:1px dashed #111d30 !important; border-radius:14px !important; }
.stTextArea textarea { background:#060b16 !important; color:#c4cfe0 !important; border:1px solid #111d30 !important; border-radius:10px !important; font-family:'Inter',sans-serif !important; font-size:14px !important; }
.stTextArea textarea:focus { border-color:#2acea840 !important; box-shadow:0 0 0 2px #2acea810 !important; }
.stMultiSelect > div { background:#060b16 !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid #111d30 !important; gap:4px; }
.stTabs [data-baseweb="tab"] { font-family:'Syne',sans-serif !important; font-size:10px !important; font-weight:700 !important; letter-spacing:3px !important; color:#253040 !important; background:transparent !important; border:none !important; text-transform:uppercase !important; padding:8px 18px !important; }
.stTabs [aria-selected="true"] { color:#2acea8 !important; border-bottom:2px solid #2acea8 !important; }
hr { border-color:#0e1826 !important; }
div[data-testid="stExpander"] { background:#070c18 !important; border:1px solid #111d30 !important; border-radius:12px !important; }
.stProgress > div > div { background:#2acea8 !important; }
.stMarkdown p   { margin:0 0 10px !important; font-size:14.5px; line-height:1.8; color:#aabacf; }
.stMarkdown ul, .stMarkdown ol { margin:2px 0 10px !important; padding-left:20px; }
.stMarkdown li  { margin-bottom:5px !important; font-size:14px; color:#aabacf; line-height:1.7; }
.stMarkdown strong { color:#dde6f0 !important; font-weight:600; }
.stMarkdown h3  { font-family:'Syne',sans-serif !important; font-size:11px !important; font-weight:700 !important; letter-spacing:3px !important; text-transform:uppercase !important; color:#2acea8 !important; margin:20px 0 10px !important; border-bottom:1px solid #0f2820; padding-bottom:6px; }
.stMarkdown h2  { font-family:'Syne',sans-serif !important; font-size:13px !important; font-weight:700 !important; color:#5ba4f5 !important; margin:16px 0 8px !important; }
.stMarkdown code { background:#0a1628 !important; border:1px solid #152240 !important; border-radius:4px !important; padding:1px 7px !important; font-size:12.5px !important; color:#5ba4f5 !important; }
.stMarkdown blockquote { border-left:2px solid #a78bfa !important; margin:8px 0 !important; padding:4px 14px !important; color:#7a8fa8 !important; font-style:italic; }
@keyframes pulse { 0%,100%{box-shadow:0 0 8px #2acea8,0 0 16px #2acea860;} 50%{box-shadow:0 0 12px #2acea8,0 0 28px #2acea8aa;} }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    st.session_state.pipeline_log.append(f"[{ts}]  {msg}")

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

# ── Sentence-transformers embedding (runs locally in Python, no Ollama) ──
@st.cache_resource
def load_sbert_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text):
    model = load_sbert_model()
    return model.encode(text).tolist()

def create_embeddings_batch(texts):
    model = load_sbert_model()
    return model.encode(texts).tolist()

# ── Groq LLM ──
def generate_response(prompt, temperature=0.4):
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set in Streamlit secrets."
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1500
        },
        timeout=30
    )
    return r.json()["choices"][0]["message"]["content"]

def check_groq_status():
    if not GROQ_API_KEY:
        return "no_key"
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=5
        )
        return "ready" if r.status_code == 200 else "error"
    except:
        return "offline"

def search(query, df, top_k):
    emb = create_embedding(query)
    matrix = np.vstack(df["embedding"].values)
    sims = cosine_similarity(matrix, [emb]).flatten()
    idx = sims.argsort()[::-1][:top_k]
    results = df.iloc[idx].copy()
    results["similarity"] = sims[idx]
    return results

@st.cache_data
def load_embeddings():
    if os.path.exists("embeddings.joblib"):
        return joblib.load("embeddings.joblib")
    return None

def stepper_html(active):
    steps = ["UPLOAD", "CONVERT", "TRANSCRIBE", "EMBED", "SEARCH"]
    nodes = []
    for i, s in enumerate(steps):
        cls  = "done" if i < active else ("active" if i == active else "")
        icon = "✓"   if i < active else str(i + 1)
        nodes.append(f'<div class="step-node"><div class="step-circle {cls}">{icon}</div><div class="step-label {cls}">{s}</div></div>')
        if i < len(steps) - 1:
            line_cls = "done" if i < active else ""
            nodes.append(f'<div class="step-line {line_cls}"></div>')
    return f'<div class="stepper-wrap">{"".join(nodes)}</div>'

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙  CONFIG")
    top_k   = st.slider("Top K Results", 1, 15, 5)
    min_sim = st.slider("Min Similarity", 0.0, 1.0, 0.2)
    st.markdown("---")
    st.markdown("### 🤖  MODEL")
    st.markdown(f"**LLM:** `{GROQ_MODEL}`")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4)

    # ── Groq status card ──
    groq_status = check_groq_status()
    if groq_status == "ready":
        st.markdown("""
        <div style="background:linear-gradient(135deg,#071912,#0a1f18);border:1px solid #1a4a38;border-radius:12px;padding:14px 16px;margin-top:12px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#2acea8,transparent);"></div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#2acea8;box-shadow:0 0 8px #2acea8,0 0 16px #2acea860;flex-shrink:0;animation:pulse 2s infinite;"></div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:2px;color:#2acea8;text-transform:uppercase;">Groq Connected</div>
                    <div style="font-size:10px;color:#1a5a45;margin-top:2px;letter-spacing:1px;">llama3-70b · Ready</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    elif groq_status == "no_key":
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a0608,#200a0c);border:1px solid #4a1018;border-radius:12px;padding:14px 16px;margin-top:12px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#f43f5e,transparent);"></div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#f43f5e;box-shadow:0 0 8px #f43f5e88;flex-shrink:0;"></div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:2px;color:#f43f5e;text-transform:uppercase;">No API Key</div>
                    <div style="font-size:10px;color:#6a1020;margin-top:2px;letter-spacing:1px;">Add GROQ_API_KEY to secrets</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1200,#201800);border:1px solid #4a3800;border-radius:12px;padding:14px 16px;margin-top:12px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#f59e0b,transparent);"></div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:10px;height:10px;border-radius:50%;background:#f59e0b;box-shadow:0 0 8px #f59e0b88;flex-shrink:0;"></div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:2px;color:#f59e0b;text-transform:uppercase;">Groq Unreachable</div>
                    <div style="font-size:10px;color:#6a4e00;margin-top:2px;letter-spacing:1px;">Check your API key</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Embedding model info ──
    st.markdown("""
    <div style="background:#080d18;border:1px solid #111d30;border-radius:12px;padding:14px 16px;margin-top:12px;">
        <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:2px;color:#5ba4f5;text-transform:uppercase;">Embedding Model</div>
        <div style="font-size:10px;color:#253040;margin-top:4px;letter-spacing:1px;">all-MiniLM-L6-v2 · Local</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊  SESSION")
    st.markdown(f"**Queries:** `{st.session_state.query_count}`")
    if st.button("Clear Logs"):
        st.session_state.pipeline_log = []
        st.rerun()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="wordmark">◈  VIDMIND AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Video Knowledge Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload · Transcribe · Embed · Discover</div>', unsafe_allow_html=True)
st.markdown(stepper_html(st.session_state.active_step), unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STEP 1 — UPLOAD VIDEOS
# ─────────────────────────────────────────────
with st.container():
    st.markdown('<div class="panel"><div class="panel-title">◈  Step 01 — Upload Videos</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop your video files here",
        type=["mp4", "mkv", "webm"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("videos", exist_ok=True)
        for f in uploaded_files:
            dest = os.path.join("videos", f.name)
            if not os.path.exists(dest):
                with open(dest, "wb") as out:
                    out.write(f.read())
                log(f"Saved → videos/{f.name}")

        all_vids = [v for v in os.listdir("videos") if v.lower().endswith((".mp4", ".mkv", ".webm"))]
        st.markdown(f"**{len(all_vids)} video(s) in queue:**")
        cols = st.columns(min(len(all_vids), 4))
        for i, v in enumerate(sorted(all_vids)):
            cols[i % 4].markdown(f'<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;padding:10px 14px;font-size:12px;color:#6eb4ff;font-family:\'Syne\',sans-serif;word-break:break-all;">🎬 {v}</div>', unsafe_allow_html=True)

        if st.session_state.active_step == 0:
            st.session_state.active_step = 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STEP 2 — CONVERT VIDEO → MP3
# ─────────────────────────────────────────────
with st.container():
    st.markdown('<div class="panel"><div class="panel-title">◈  Step 02 — Convert Video → Audio</div>', unsafe_allow_html=True)

    videos_exist = os.path.exists("videos") and any(f.lower().endswith((".mp4",".mkv",".webm")) for f in os.listdir("videos"))
    audios_exist = os.path.exists("audios") and any(f.endswith(".mp3") for f in os.listdir("audios"))

    c1, c2 = st.columns([3, 1])
    with c1:
        if audios_exist:
            count = len([f for f in os.listdir("audios") if f.endswith(".mp3")])
            st.markdown(f'<span class="dot-ok">● {count} audio file(s) already converted</span>', unsafe_allow_html=True)
        else:
            st.markdown("Convert all uploaded videos to `.mp3` using FFmpeg.")
    with c2:
        convert_btn = st.button("Convert to MP3", disabled=not videos_exist, use_container_width=True)

    if convert_btn:
        os.makedirs("audios", exist_ok=True)
        video_files = sorted([f for f in os.listdir("videos") if f.lower().endswith((".mp4",".mkv",".webm"))])
        progress = st.progress(0, text="Converting…")
        for i, file in enumerate(video_files, start=1):
            tutorial_number = f"{i:02d}"
            file_name = os.path.splitext(file)[0]
            input_path = os.path.join("videos", file)
            mp3_path   = os.path.join("audios", f"{tutorial_number}_{file_name}.mp3")
            log(f"Converting {file} → {mp3_path}")
            try:
                subprocess.run(["ffmpeg","-y","-i",input_path,"-vn","-ab","192k","-ar","44100",mp3_path], check=True, capture_output=True)
                log(f"✓ Done: {mp3_path}")
            except subprocess.CalledProcessError as e:
                log(f"✗ Failed: {file} — {e}")
            progress.progress(i / len(video_files), text=f"Converted {i}/{len(video_files)}")
        progress.empty()
        st.success("All videos converted to MP3!")
        st.session_state.active_step = max(st.session_state.active_step, 2)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STEP 3 — TRANSCRIBE MP3 → JSON
# ─────────────────────────────────────────────
with st.container():
    st.markdown('<div class="panel"><div class="panel-title">◈  Step 03 — Transcribe Audio → Text (Whisper)</div>', unsafe_allow_html=True)

    jsons_exist  = os.path.exists("jsons") and any(f.endswith(".json") for f in os.listdir("jsons")) if os.path.exists("jsons") else False
    audios_ready = os.path.exists("audios") and any(f.endswith(".mp3") for f in os.listdir("audios")) if os.path.exists("audios") else False

    c1, c2 = st.columns([3, 1])
    with c1:
        if jsons_exist:
            count = len([f for f in os.listdir("jsons") if f.endswith(".json")])
            st.markdown(f'<span class="dot-ok">● {count} transcript JSON(s) ready</span>', unsafe_allow_html=True)
        else:
            st.markdown("Use OpenAI Whisper to transcribe each audio file to timestamped JSON.")
    with c2:
        transcribe_btn = st.button("Transcribe", disabled=not audios_ready, use_container_width=True)

    if transcribe_btn:
        try:
            import whisper as _whisper
        except ImportError:
            st.error("openai-whisper not installed.")
            st.stop()

        os.makedirs("jsons", exist_ok=True)
        audio_list = [f for f in os.listdir("audios") if "_" in f and f.endswith(".mp3")]
        if not audio_list:
            st.warning("No MP3 files found. Run Step 2 first.")
        else:
            with st.spinner("Loading Whisper model…"):
                wmodel = _whisper.load_model("base")
            progress = st.progress(0, text="Transcribing…")
            for i, audio in enumerate(audio_list, start=1):
                number = audio.split("_")[0]
                title  = audio.split("_")[1][:-4]
                log(f"Transcribing: {audio}")
                try:
                    result = wmodel.transcribe(audio=f"audios/{audio}", task="transcribe")
                    chunks = []
                    for seg in result["segments"]:
                        text = clean_text(seg["text"])
                        if len(text) < 5:
                            continue
                        chunks.append({"number": number, "title": title, "start": seg["start"], "end": seg["end"], "text": text})
                    payload = {"chunks": chunks, "text": clean_text(result["text"])}
                    with open(f"jsons/{audio}.json", "w", encoding="utf-8") as jf:
                        json.dump(payload, jf, indent=4, ensure_ascii=False)
                    log(f"✓ Saved jsons/{audio}.json ({len(chunks)} chunks)")
                except Exception as e:
                    log(f"✗ Error on {audio}: {e}")
                progress.progress(i / len(audio_list), text=f"Transcribed {i}/{len(audio_list)}")
            progress.empty()
            st.success("Transcription complete!")
            st.session_state.active_step = max(st.session_state.active_step, 3)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STEP 4 — EMBED → SAVE
# ─────────────────────────────────────────────
with st.container():
    st.markdown('<div class="panel"><div class="panel-title">◈  Step 04 — Generate Embeddings (all-MiniLM-L6-v2)</div>', unsafe_allow_html=True)

    emb_exists   = os.path.exists("embeddings.joblib")
    jsons_ready  = os.path.exists("jsons") and any(f.endswith(".json") for f in os.listdir("jsons")) if os.path.exists("jsons") else False

    c1, c2 = st.columns([3, 1])
    with c1:
        if emb_exists:
            st.markdown('<span class="dot-ok">● embeddings.joblib exists</span>', unsafe_allow_html=True)
        else:
            st.markdown("Creates vector embeddings for every transcript chunk using sentence-transformers.")
    with c2:
        embed_btn = st.button("Build Embeddings", disabled=not jsons_ready, use_container_width=True)

    if embed_btn:
        import pandas as _pd

        json_files = [f for f in os.listdir("jsons") if f.endswith(".json")]
        my_dicts   = []
        chunk_id   = 0
        errors     = 0

        with st.spinner("Loading sentence-transformers model…"):
            sbert = load_sbert_model()

        progress = st.progress(0, text="Embedding…")
        for fi, json_file in enumerate(json_files, start=1):
            with open(f"jsons/{json_file}", encoding="utf-8") as jf:
                content = json.load(jf)
            log(f"Embedding: {json_file}")

            texts, valid_chunks = [], []
            for chunk in content.get("chunks", []):
                t = clean_text(chunk["text"])
                if len(t) < 10:
                    continue
                texts.append(t)
                valid_chunks.append(chunk)

            if not texts:
                continue

            try:
                embeddings = sbert.encode(texts).tolist()
            except Exception as e:
                log(f"✗ Embedding error on {json_file}: {e}")
                errors += 1
                continue

            for i, chunk in enumerate(valid_chunks):
                ct = clean_text(chunk["text"])
                my_dicts.append({
                    "chunk_id":  chunk_id,
                    "number":    chunk.get("number"),
                    "title":     chunk.get("title"),
                    "start":     chunk.get("start"),
                    "end":       chunk.get("end"),
                    "text":      ct,
                    "embedding": embeddings[i],
                    "preview":   ct[:80],
                    "topic":     chunk.get("title", "").replace("-", " ")
                })
                chunk_id += 1

            progress.progress(fi / len(json_files), text=f"Embedded {fi}/{len(json_files)}")

        if my_dicts:
            emb_df = _pd.DataFrame.from_records(my_dicts)
            joblib.dump(emb_df, "embeddings.joblib")
            log(f"✓ Saved embeddings.joblib ({len(emb_df)} chunks)")
            st.success(f"Embeddings built! {len(emb_df)} chunks indexed.")
            st.cache_data.clear()
            st.session_state.active_step = max(st.session_state.active_step, 4)
        else:
            st.error("No chunks were embedded. Check logs.")

        progress.empty()
        if errors:
            st.warning(f"{errors} file(s) had errors — see logs.")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PIPELINE LOG
# ─────────────────────────────────────────────
if st.session_state.pipeline_log:
    with st.expander("Pipeline Log", expanded=False):
        log_html = "<br>".join(st.session_state.pipeline_log[-60:])
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  STEP 5 — SEARCH
# ─────────────────────────────────────────────
df = load_embeddings()

if df is None:
    st.markdown("""
    <div style="text-align:center;padding:48px;color:#2e3a50;">
        <div style="font-family:'Syne',sans-serif;font-size:32px;margin-bottom:12px;">⬆</div>
        <div style="font-family:'Syne',sans-serif;font-size:13px;letter-spacing:2px;color:#2acea8;">Complete the pipeline above to unlock search</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="panel-title">◈  KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="chips-row">
        <div class="chip"><div class="chip-val">{len(df)}</div><div class="chip-lbl">Total Chunks</div></div>
        <div class="chip"><div class="chip-val">{df["title"].nunique()}</div><div class="chip-lbl">Videos Indexed</div></div>
        <div class="chip"><div class="chip-val">{int(df["text"].str.len().mean())}</div><div class="chip-lbl">Avg Chunk Len</div></div>
        <div class="chip"><div class="chip-val">{int(len(df) * 5)}</div><div class="chip-lbl">Est. Seconds</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel-title" style="margin-top:32px;">◈  Step 05 — Search &amp; Ask</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["NORMAL SEARCH", "ADVANCED SEARCH"])

    with tab1:
        query = st.text_area("Ask anything from your videos…", key="query_input",
                             placeholder="e.g. What is a neural network?", height=100)
        c1, c2 = st.columns([1, 1])
        search_btn = c1.button("Search", use_container_width=True)
        def clear_q():
            st.session_state.query_input = ""
        c2.button("Clear", on_click=clear_q, use_container_width=True)

    with tab2:
        filter_video = st.multiselect("Filter by Video", df["title"].unique())
        adv_query = st.text_area("Advanced Query", key="adv_query_input", height=80)
        c1, c2 = st.columns([1, 1])
        adv_btn = c1.button("Filter Search", use_container_width=True)
        def clear_adv():
            st.session_state.adv_query_input = ""
            st.session_state.last_results = None
            st.session_state.ai_response  = ""
        c2.button("Clear All", on_click=clear_adv, use_container_width=True)

    final_query = query if (search_btn and query) else (adv_query if (adv_btn and adv_query) else None)

    if final_query:
        st.session_state.query_count += 1
        st.session_state.active_step  = max(st.session_state.active_step, 4)

        temp_df = df.copy()
        if adv_btn and filter_video:
            temp_df = temp_df[temp_df["title"].isin(filter_video)]

        with st.spinner("Searching knowledge base…"):
            results = search(final_query, temp_df, top_k)
            results = results[results["similarity"] >= min_sim]

        if results.empty:
            st.warning("No results found above the similarity threshold.")
        else:
            st.session_state.last_results = results

            st.markdown(f'<div class="panel-title" style="margin-top:28px;">◈  RELEVANT CHUNKS  ({len(results)} found)</div>', unsafe_allow_html=True)

            for _, row in results.iterrows():
                start = row.get("start")
                end   = row.get("end")
                ts = f"{int(start)//60:02d}:{int(start)%60:02d} → {int(end)//60:02d}:{int(end)%60:02d}" if start is not None and end is not None else "N/A"
                st.markdown(f"""
                <div class="res-card">
                    <div class="res-header">
                        <div class="res-title">{row['title'][:70]}</div>
                        <div class="res-score">{row['similarity']:.3f}</div>
                    </div>
                    <div class="res-time">⏱ {ts}</div>
                    <div class="res-text">{row['text']}</div>
                </div>""", unsafe_allow_html=True)

            context = results[["title", "start", "end", "text"]].to_json(orient="records")
            PROMPT = f"""You are an expert AI Teaching Assistant. Your goal is to TEACH the student thoroughly.

STRICT RULES:
- Use ONLY the provided context. Never invent facts outside it.
- Write at least 150-250 words minimum.
- Structure your answer clearly with these sections:

### 📖 Explanation
Write 3-5 paragraphs explaining the concept deeply in simple, plain English. No bullet walls — use flowing prose.

### 💡 Real-World Example
Give at least ONE concrete, relatable real-world analogy or example that makes the concept click for a beginner.

### 🔑 Key Takeaways
3-5 crisp bullet points summarising what the student must remember.

### 🎬 Where to Learn More
List each relevant video chunk with its title and timestamp:
- **[Video Title]** → ⏱ MM:SS - MM:SS

FORMATTING RULES:
- Use **bold** for important terms when first introduced.
- Keep paragraphs tight (3-4 sentences max).
- If the context doesn't cover the question, say: "I couldn't find a detailed explanation in the uploaded videos."

Context:
{context}

Question:
{final_query}
"""
            with st.spinner("Generating AI response…"):
                response = generate_response(PROMPT, temperature)
            st.session_state.ai_response = response

            st.markdown('<div class="panel-title" style="margin-top:28px;">◈  AI TEACHING RESPONSE</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown(response)

    if st.session_state.last_results is not None:
        st.markdown('<div class="panel-title" style="margin-top:28px;">◈  EXPORT</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.download_button("AI Response (.txt)", st.session_state.ai_response, file_name="ai_response.txt")
        col2.download_button("Results (.csv)", st.session_state.last_results.to_csv(index=False), file_name="results.csv")
        report = f"Query: {final_query or ''}\n\n{st.session_state.ai_response}"
        col3.download_button("Full Report (.txt)", report, file_name="report.txt")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:60px 0 30px;'>
    <div style='font-family:"Syne",sans-serif;font-size:9px;letter-spacing:5px;color:#0e1826;text-transform:uppercase;font-weight:700;'>
        VIDMIND AI  ·  GROQ + WHISPER + SENTENCE-TRANSFORMERS  ·  © 2026
    </div>
</div>
""", unsafe_allow_html=True)
