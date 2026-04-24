import os
import sys
import subprocess
import base64
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from summary import summarize_segments, fetch_google_images, generate_comprehensive_summary

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except ImportError:
    HashingVectorizer = None

DEFAULT_VIDEO_PATH = "videos/input.mp4"
OUTPUT_DIR = "output"
AUDIO_FILENAME = "extracted_audio.wav"
DEFAULT_TOP_K = 5
CACHE_FILE = os.path.join(OUTPUT_DIR, "prepared_cache.pkl")  # Persistent cache

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import whisper
except ImportError:
    whisper = None

SENTENCE_TRANSFORMER_IMPORT_ERROR = None
try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None
    SENTENCE_TRANSFORMER_IMPORT_ERROR = str(exc)

try:
    import faiss
except ImportError:
    faiss = None

EMOTION_CLASSIFIER = None
EMOTION_CLASSIFIER_INITIALIZED = False

def load_emotion_classifier():
    """Lazy load emotion classifier on first use."""
    global EMOTION_CLASSIFIER, EMOTION_CLASSIFIER_INITIALIZED
    
    if EMOTION_CLASSIFIER_INITIALIZED:
        return EMOTION_CLASSIFIER
    
    EMOTION_CLASSIFIER_INITIALIZED = True
    
    try:
        from transformers import pipeline
        print("Loading zero-shot emotion classification model...")
        # Use zero-shot classification for better performance
        EMOTION_CLASSIFIER = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # Use CPU
        )
        print("[OK] Emotion classifier loaded successfully!")
        return EMOTION_CLASSIFIER
    except ImportError:
        print("[ERROR] transformers not installed. Install with: pip install transformers torch")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading emotion classifier: {e}")
        return None


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_cache(video_path: str, segments, embeddings, model):
    """Save prepared video data to disk for persistence across sessions."""
    ensure_output_dir()
    try:
        backend_name = getattr(model, "backend_name", None) if model is not None else None
        cache_data = {
            "video_path": video_path,
            "segments": segments,
            "embeddings": embeddings,
            "backend_name": backend_name,
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def load_cache():
    """Load prepared video data from disk if it exists."""
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    return None


def clear_cache():
    """Clear the persistent cache."""
    if os.path.isfile(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")


def load_logo_image_data():
    candidates = [
        "src/logo.png",
        "src/logo.jpg",
        "src/logo.jpeg",
        "src/logo.gif",
        "src/logo.svg",
        "logo.png",
        "logo.jpg",
        "logo.jpeg",
        "logo.gif",
        "logo.svg",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            data = open(candidate, "rb").read()
            ext = Path(candidate).suffix.lower()
            if ext == ".svg":
                mime = "image/svg+xml"
            elif ext == ".png":
                mime = "image/png"
            elif ext in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif ext == ".gif":
                mime = "image/gif"
            else:
                continue
            return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"
    return None


def build_background_particles_html(count: int = 48):
    particles = []
    for idx in range(count):
        left = (idx * 7.3) % 100
        size = 2 + (idx % 5) * 2
        duration = 14 + (idx % 9) * 2
        delay = (idx % 8) * 1.3
        opacity = 0.18 + (idx % 4) * 0.08
        blur = (idx % 3) * 1.2
        particles.append(
            (
                "<span class='bg-particle' "
                f"style='left:{left:.2f}%;width:{size}px;height:{size}px;"
                f"--particle-duration:{duration}s;--particle-delay:{delay:.1f}s;"
                f"--particle-opacity:{opacity:.2f};--particle-blur:{blur:.1f}px;'></span>"
            )
        )
    return "<div class='background-particles'>" + "".join(particles) + "</div>"


def extract_audio(video_path: str, audio_path: str):
    ensure_output_dir()
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to extract audio:\n{result.stderr.strip()}"
        )
    return audio_path


def load_whisper_model(model_name: str = "small"):
    if whisper is None:
        raise ImportError("whisper is not installed. Install openai-whisper first.")
    return whisper.load_model(model_name)


def transcribe_audio(audio_path: str, model=None):
    if model is None:
        model = load_whisper_model()
    result = model.transcribe(audio_path)
    return [
        {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
        for seg in result.get("segments", [])
    ]


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        details = (
            f" Details: {SENTENCE_TRANSFORMER_IMPORT_ERROR}"
            if SENTENCE_TRANSFORMER_IMPORT_ERROR
            else ""
        )
        raise ImportError(f"sentence-transformers is unavailable.{details}")
    return SentenceTransformer(model_name)


class FallbackTextEncoder:
    def __init__(self):
        if HashingVectorizer is None:
            raise ImportError("scikit-learn is unavailable, so fallback embeddings cannot be created.")
        self.vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm="l2",
        )
        self.backend_name = "hashing"

    def encode(self, texts, convert_to_tensor=False):
        if isinstance(texts, str):
            texts = [texts]
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


def load_text_encoder(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is not None:
        model = load_sentence_transformer(model_name)
        model.backend_name = "sentence-transformers"
        return model
    return FallbackTextEncoder()


def load_cached_model(backend_name: Optional[str] = None):
    if backend_name == "sentence-transformers":
        try:
            return load_text_encoder()
        except Exception:
            return FallbackTextEncoder()
    if backend_name == "hashing":
        return FallbackTextEncoder()
    try:
        return load_text_encoder()
    except Exception:
        return None


def create_embeddings(segments):
    # Detect emotions for all segments first
    segments = detect_all_segment_emotions(segments)
    
    # Create text embeddings
    model = load_text_encoder()
    texts = [segment["text"] for segment in segments]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings, dtype=np.float32), model


def build_index(embeddings: np.ndarray):
    if faiss is None:
        raise ImportError("faiss-cpu is not installed.")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-10, None)
    index.add(normalized)
    return index


def normalize_embeddings(embeddings: np.ndarray):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-10, None)


EMOTION_LABELS = ["happy", "sad", "fear", "disgust", "awful"]
EMOTION_THRESHOLD = 0.45


def find_segments(query: str, segments):
    query_text = query.strip().lower()
    if not query_text:
        return []
    return [
        segment
        for segment in segments
        if query_text in segment["text"].lower()
    ]


def detect_emotion(query: str, model):
    if model is None:
        return None, 0.0
    if not query.strip():
        return None, 0.0

    query_vec = model.encode([query], convert_to_tensor=False)
    label_vecs = model.encode(EMOTION_LABELS, convert_to_tensor=False)
    query_vec = np.asarray(query_vec, dtype=np.float32)
    label_vecs = np.asarray(label_vecs, dtype=np.float32)

    query_norm = query_vec / np.clip(np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-10, None)
    label_norm = label_vecs / np.clip(np.linalg.norm(label_vecs, axis=1, keepdims=True), 1e-10, None)

    scores = label_norm.dot(query_norm.T).flatten()
    best_index = int(np.argmax(scores))
    return EMOTION_LABELS[best_index], float(scores[best_index])


def find_emotional_segments(emotion: str, segments, model, top_k: int = DEFAULT_TOP_K):
    if model is None:
        raise ImportError("sentence-transformers model is not loaded.")
    if not emotion or not segments:
        return []

    texts = [segment["text"] for segment in segments]
    segment_embeddings = model.encode(texts, convert_to_tensor=False)
    emotion_vec = model.encode([emotion], convert_to_tensor=False)

    segment_embeddings = np.asarray(segment_embeddings, dtype=np.float32)
    emotion_vec = np.asarray(emotion_vec, dtype=np.float32)

    segment_norm = segment_embeddings / np.clip(np.linalg.norm(segment_embeddings, axis=1, keepdims=True), 1e-10, None)
    emotion_norm = emotion_vec / np.clip(np.linalg.norm(emotion_vec, axis=1, keepdims=True), 1e-10, None)

    scores = segment_norm.dot(emotion_norm.T).flatten()
    best_indices = np.argsort(-scores)[:top_k]
    return [
        {**segments[i], "score": float(scores[i])}
        for i in best_indices
    ]


def classify_segment_emotions(text: str):
    """Classify emotions in a text segment using zero-shot classification."""
    classifier = load_emotion_classifier()
    if classifier is None:
        return {}
    try:
        # Define emotion candidates
        candidate_emotions = ["happy", "sad", "angry", "scared", "excited", "confused", "neutral", "disgusted", "surprised"]
        
        # Run zero-shot classification
        result = classifier(text[:512], candidate_emotions)  # Limit to 512 chars
        
        # Convert to emotion scores
        emotion_scores = {}
        if "scores" in result and "labels" in result:
            for label, score in zip(result["labels"], result["scores"]):
                emotion_scores[label.lower()] = float(score)
        
        return emotion_scores
    except Exception as e:
        print(f"Error classifying emotions: {e}")
        return {}


def detect_all_segment_emotions(segments):
    """Detect emotions for all segments and add to segment data."""
    classifier = load_emotion_classifier()
    if classifier is None:
        # If classifier unavailable, add empty emotions dict
        for segment in segments:
            segment["emotions"] = {}
        return segments
    
    try:
        for i, segment in enumerate(segments):
            emotions = classify_segment_emotions(segment.get("text", ""))
            segment["emotions"] = emotions
            if (i + 1) % 5 == 0:
                print(f"Emotions detected for {i + 1}/{len(segments)} segments")
    except Exception as e:
        print(f"Error detecting segment emotions: {e}")
        for segment in segments:
            if "emotions" not in segment:
                segment["emotions"] = {}
    
    return segments


def find_segments_by_emotion_classifier(target_emotion: str, segments, top_k: int = DEFAULT_TOP_K):
    """Find segments that match target emotion using pre-trained classifier."""
    classifier = load_emotion_classifier()
    if classifier is None:
        return []
    
    target_emotion_lower = target_emotion.lower().strip()
    scored_segments = []
    
    # Check each segment's detected emotions
    for segment in segments:
        emotions = segment.get("emotions", {})
        
        # Direct match: look for exact emotion in detected emotions
        if target_emotion_lower in emotions:
            score = emotions[target_emotion_lower]
        else:
            # Fallback: run zero-shot classification on the segment text for this specific emotion
            try:
                result = classifier(segment.get("text", "")[:512], [target_emotion_lower])
                if "scores" in result and len(result["scores"]) > 0:
                    score = float(result["scores"][0])
                else:
                    score = 0.0
            except Exception:
                score = 0.0
        
        if score > 0.0:  # Only include segments with detected emotion
            scored_segments.append({**segment, "emotion_score": score})
    
    # Sort by emotion score and return top_k
    sorted_segments = sorted(scored_segments, key=lambda x: x.get("emotion_score", 0.0), reverse=True)
    return sorted_segments[:top_k]


def search_segments(query: str, embeddings, model, segments, top_k: int = DEFAULT_TOP_K):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed.")

    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)
    elif isinstance(embeddings, np.ndarray):
        embeddings = embeddings.astype(np.float32)

    query_model = model if model is not None else load_text_encoder()
    query_vec = query_model.encode([query], convert_to_tensor=False)
    query_vec = np.asarray(query_vec, dtype=np.float32)
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    embeddings_normed = normalize_embeddings(embeddings)
    query_norm = query_vec / np.clip(np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-10, None)
    scores = embeddings_normed.dot(query_norm.T).flatten()
    best = np.argsort(-scores)[:top_k]

    return [
        {**segments[i], "score": float(scores[i])}
        for i in best
    ]


def play_video(timestamp: float, video_path: str = DEFAULT_VIDEO_PATH):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    command = ["ffplay", "-ss", str(timestamp), "-autoexit", video_path]
    subprocess.Popen(command, shell=False)


def render_video_player(video_path: str, start_time: float = 0.0, height: int = 480, autoplay: bool = False):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    autoplay_code = "video.play();" if autoplay else ""
    html_code = f"""
    <style>
      .streamlit-video-wrapper {{
        max-width: 100%;
      }}
      video {{
        width: 100%;
        height: auto;
        max-height: 80vh;
      }}
    </style>
    <div class='streamlit-video-wrapper'>
      <video id='video-player' controls preload='metadata'>
        <source src='data:video/mp4;base64,{video_b64}' type='video/mp4'>
        Your browser does not support HTML5 video.
      </video>
    </div>
    <script>
      const video = document.getElementById('video-player');
      video.addEventListener('loadedmetadata', function() {{
        if (video.readyState >= 1) {{
          video.currentTime = {start_time};
          {autoplay_code}
        }}
      }});
    </script>
    """
    st.components.v1.html(html_code, height=height, scrolling=False)


if st is not None:
    @st.dialog("Generated Summary", dismissible=True, width="large")
    def show_summary_dialog(summary_text: str):
        st.markdown(
            """
            <style>
            [data-testid="stDialog"] {
              backdrop-filter: blur(16px) saturate(135%);
              -webkit-backdrop-filter: blur(16px) saturate(135%);
            }
            [data-testid="stDialog"] [role="dialog"] {
              background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.05)),
                linear-gradient(145deg, rgba(13, 20, 38, 0.9), rgba(7, 10, 22, 0.96));
              border: 1px solid rgba(255, 255, 255, 0.14);
              box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.14),
                0 34px 100px rgba(0, 0, 0, 0.45),
                0 0 0 1px rgba(255,255,255,0.03);
              backdrop-filter: blur(28px) saturate(160%);
              -webkit-backdrop-filter: blur(28px) saturate(160%);
              border-radius: 32px;
            }
            [data-testid="stDialog"] [role="dialog"]::before {
              content: "";
              position: absolute;
              inset: 0;
              border-radius: 32px;
              background:
                radial-gradient(circle at top, rgba(96, 165, 250, 0.16), transparent 30%),
                radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.2), transparent 38%);
              pointer-events: none;
            }
            [data-testid="stDialog"] [role="dialog"] h2 {
              color: #ffffff;
              font-weight: 900;
              letter-spacing: -0.03em;
            }
            [data-testid="stDialog"] [role="dialog"] p,
            [data-testid="stDialog"] [role="dialog"] div {
              color: #e2e8f0;
            }
            [data-testid="stDialog"] [role="dialog"] button[kind="secondary"] {
              background: rgba(255,255,255,0.08);
              border: 1px solid rgba(255,255,255,0.10);
              box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span style='display:inline-block;padding:8px 14px;border-radius:999px;background:rgba(139,92,246,0.16);color:#d8b4fe;font-size:0.83rem;font-weight:700;letter-spacing:0.03em;margin-bottom:12px;'>Video Summary</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="line-height:1.9;font-size:1rem;color:#e2e8f0;white-space:pre-wrap;">
            {summary_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}
            </div>
            <div style="margin-top:18px;color:#94a3b8;font-size:0.92rem;">
              Summary length: {len(summary_text)} characters
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Close Summary", use_container_width=True):
            st.rerun()

    @st.dialog("Search Results", dismissible=True, width="large")
    def show_results_dialog(results):
        st.markdown(
            """
            <style>
            [data-testid="stDialog"] {
              backdrop-filter: blur(16px) saturate(135%);
              -webkit-backdrop-filter: blur(16px) saturate(135%);
            }
            [data-testid="stDialog"] [role="dialog"] {
              background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.05)),
                linear-gradient(145deg, rgba(13, 20, 38, 0.9), rgba(7, 10, 22, 0.96)) !important;
              border: 1px solid rgba(255, 255, 255, 0.14) !important;
              box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.14),
                0 34px 100px rgba(0, 0, 0, 0.45) !important;
              backdrop-filter: blur(28px) saturate(160%);
              -webkit-backdrop-filter: blur(28px) saturate(160%);
              border-radius: 32px !important;
              max-height: 80vh;
              overflow-y: auto;
            }
            .dialog-result-item {
              margin-bottom: 20px;
              padding: 24px;
              border-radius: 28px;
              background: rgba(255,255,255,0.06);
              border: 1px solid rgba(255,255,255,0.1);
              transition: all 0.3s ease;
            }
            .dialog-result-item:hover {
              background: rgba(255,255,255,0.1);
              border-color: rgba(125, 211, 252, 0.4);
              transform: translateY(-2px);
            }
            .result-timestamp {
              font-size: 0.88rem;
              font-weight: 850;
              color: #7dd3fc;
              text-transform: uppercase;
              margin-bottom: 10px;
              display: block;
              letter-spacing: 0.02em;
            }
            .result-text {
              color: #f1f5f9;
              line-height: 1.7;
              font-size: 1.05rem;
              margin-bottom: 18px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        if not results:
            st.info("No results found for your query.")
            return

        for i, res in enumerate(results):
            with st.container():
                st.markdown(
                    f'<div class="dialog-result-item">'
                    f'<span class="result-timestamp">Segment {i+1} • {res["start"]}s</span>'
                    f'<p class="result-text">{res["text"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"▶ Play this segment", key=f"play_{i}"):
                    st.session_state.selected_start = float(res["start"])
                    st.session_state.autoplay_segment = True
                    st.session_state.show_results = False
                    st.rerun()
        
        if st.button("Close Results", use_container_width=True):
            st.session_state.show_results = False
            st.rerun()
else:
    def show_summary_dialog(summary_text: str):
        return None


def run_cli():
    print("Video Search CLI")
    print("-----------------")
    print("Use Streamlit for the UI: streamlit run src/main.py")

    video_path = DEFAULT_VIDEO_PATH
    if not os.path.isfile(video_path):
        video_path = input("Enter the path to the video file: ").strip()

    audio_path = os.path.join(OUTPUT_DIR, AUDIO_FILENAME)
    print(f"Extracting audio from {video_path}...")
    extract_audio(video_path, audio_path)
    whisper_model = load_whisper_model()
    segments = transcribe_audio(audio_path, model=whisper_model)
    if not segments:
        raise RuntimeError("No transcription segments were produced.")

    embeddings, sentence_model = create_embeddings(segments)
    build_index(embeddings)

    while True:
        query = input("\nEnter search query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = search_segments(query, embeddings, sentence_model, segments)
        print("\nTop Results:")
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. [{result['start']:.2f}s] {result['text']}")

        choice_text = input("Select result number to play or press Enter to skip: ").strip()
        if choice_text.isdigit():
            choice = int(choice_text) - 1
            if 0 <= choice < len(results):
                play_video(results[choice]["start"], video_path)


def streamlit_app():
    if st is None:
        raise ImportError("streamlit is not installed. Install streamlit to use the UI.")

    st.set_page_config(page_title="Video Search App", layout="wide")
    logo_data = load_logo_image_data()

    logo_html = ""
    if logo_data:
        logo_html = f'<img class="hero-logo-image" src="{logo_data}" alt="logo" />'
    else:
        logo_html = """
          <svg viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="petalGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#8b5cf6" stop-opacity="0.9" />
                <stop offset="60%" stop-color="#6d28d9" stop-opacity="0.82" />
                <stop offset="100%" stop-color="#2e1067" stop-opacity="0.72" />
              </linearGradient>
              <linearGradient id="shadowGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#111827" stop-opacity="0.4" />
                <stop offset="100%" stop-color="#0b1224" stop-opacity="0.4" />
              </linearGradient>
            </defs>
            <g opacity="0.92">
              <ellipse cx="110" cy="36" rx="32" ry="46" fill="url(#petalGradient)" transform="rotate(-10 110 110)" />
              <ellipse cx="164" cy="85" rx="32" ry="46" fill="url(#petalGradient)" transform="rotate(40 164 85)" />
              <ellipse cx="143" cy="166" rx="32" ry="46" fill="url(#petalGradient)" transform="rotate(100 143 166)" />
              <ellipse cx="77" cy="166" rx="32" ry="46" fill="url(#petalGradient)" transform="rotate(160 77 166)" />
              <ellipse cx="32" cy="85" rx="32" ry="46" fill="url(#petalGradient)" transform="rotate(220 32 85)" />
              <ellipse cx="82" cy="34" rx="32" ry="46" fill="url(#petalGradient)" transform="rotate(280 82 34)" />
            </g>
            <g opacity="0.55">
              <ellipse cx="110" cy="32" rx="36" ry="50" fill="url(#shadowGradient)" transform="rotate(-10 110 110)" />
              <ellipse cx="160" cy="82" rx="36" ry="50" fill="url(#shadowGradient)" transform="rotate(40 160 82)" />
              <ellipse cx="140" cy="168" rx="36" ry="50" fill="url(#shadowGradient)" transform="rotate(100 140 168)" />
              <ellipse cx="80" cy="168" rx="36" ry="50" fill="url(#shadowGradient)" transform="rotate(160 80 168)" />
              <ellipse cx="38" cy="82" rx="36" ry="50" fill="url(#shadowGradient)" transform="rotate(220 38 82)" />
              <ellipse cx="88" cy="32" rx="36" ry="50" fill="url(#shadowGradient)" transform="rotate(280 88 32)" />
            </g>
            <rect x="82" y="80" width="56" height="56" rx="18" fill="rgba(15, 23, 42, 0.65)" stroke="rgba(139, 92, 246, 0.16)" stroke-width="1.8" />
            <polygon points="98,86 98,134 136,110" fill="rgba(255,255,255,0.92)" />
            <polygon points="102,90 102,130 130,110" fill="#0f172a" opacity="0.1" />
            <path d="M92 72 L126 72 L126 80" stroke="#c4b5fd" stroke-width="3" stroke-linecap="round" opacity="0.45" />
            <path d="M78 72 L52 72 L52 80" stroke="#c4b5fd" stroke-width="3" stroke-linecap="round" opacity="0.45" />
          </svg>
        """

    hero_html = """
        <style>
        :root {
          --bg-base: #03030a;
          --bg-surface: rgba(10, 14, 28, 0.74);
          --bg-surface-strong: rgba(12, 18, 34, 0.9);
          --line-soft: rgba(255, 255, 255, 0.08);
          --line-strong: rgba(255, 255, 255, 0.14);
          --text-main: #f8fafc;
          --text-soft: #cbd5e1;
          --accent-a: #c084fc;
          --accent-b: #7dd3fc;
          --accent-c: #86efac;
        }
        html, body {
          scroll-behavior: smooth;
          margin: 0;
          min-height: 100%;
          width: 100vw;
          background:
            radial-gradient(circle at top, rgba(96, 165, 250, 0.08), transparent 24%),
            radial-gradient(circle at 85% 20%, rgba(168, 85, 247, 0.12), transparent 18%),
            var(--bg-base);
          scroll-padding-top: 36px;
        }
        .hero-screen {
          position: relative;
          width: 100vw;
          height: 100vh;
          min-height: 100vh;
          display: flex;
          align-items: flex-start;
          justify-content: center;
          overflow: hidden;
          background:
            radial-gradient(circle at top center, rgba(79, 70, 229, 0.18), transparent 25%),
            radial-gradient(circle at 20% 15%, rgba(168, 85, 247, 0.16), transparent 12%),
            radial-gradient(circle at 82% 20%, rgba(99, 102, 241, 0.12), transparent 14%),
            linear-gradient(180deg, rgba(3, 3, 10, 0.92), rgba(3, 3, 10, 0.98)),
            #03030a;
        }
        .hero-screen::before {
          content: "";
          position: absolute;
          inset: 0;
          background: radial-gradient(circle at 8% 12%, rgba(168,85,247,0.22), transparent 10%),
                      radial-gradient(circle at 88% 18%, rgba(139,92,246,0.16), transparent 9%),
                      radial-gradient(circle at 50% 70%, rgba(79,70,229,0.10), transparent 16%);
          pointer-events: none;
          filter: blur(3px);
        }
        .hero-section {
          position: relative;
          z-index: 2;
          width: 100%;
          max-width: 1220px;
          padding: 42px 24px 0;
        }
        .hero-inner {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: flex-start;
          gap: 16px;
          min-height: calc(100vh - 40px);
          padding-top: 60px;
          text-align: center;
        }
        .hero-kicker {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 11px 18px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.05);
          color: #dbeafe;
          font-size: 0.88rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          backdrop-filter: blur(18px);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .hero-kicker::before {
          content: "";
          width: 10px;
          height: 10px;
          border-radius: 999px;
          background: linear-gradient(135deg, var(--accent-b), var(--accent-a));
          box-shadow: 0 0 18px rgba(125, 211, 252, 0.75);
        }
        .hero-copy {
          margin: 0;
          max-width: 760px;
          color: #cbd5e1;
          font-size: 1.08rem;
          line-height: 1.9;
          letter-spacing: 0.02em;
        }
        .hero-features {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 18px;
          margin-top: 32px;
          width: 100%;
        }
        .hero-feature {
          padding: 20px 22px;
          border-radius: 26px;
          background:
            linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)),
            rgba(10, 14, 28, 0.72);
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 22px 65px rgba(0, 0, 0, 0.22);
          backdrop-filter: blur(22px);
          text-align: left;
          transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
        }
        .hero-feature:hover {
          transform: translateY(-6px);
          border-color: rgba(192, 132, 252, 0.34);
          box-shadow: 0 28px 75px rgba(76, 29, 149, 0.22);
        }
        .hero-feature-title {
          margin: 0 0 10px;
          color: #f8fafc;
          font-size: 1rem;
          font-weight: 700;
        }
        .hero-feature-copy {
          margin: 0;
          color: #dbeafe;
          font-size: 0.92rem;
          line-height: 1.7;
        }
        .hero-logo {
          margin-bottom: 20px;
          width: 180px;
          max-width: 100%;
        }
        .hero-logo img {
          width: 100%;
          height: auto;
          display: block;
        }
        .hero-title {
          margin: 0;
          max-width: 980px;
          font-size: clamp(4.2rem, 8.5vw, 7rem);
          line-height: 0.92;
          font-weight: 900;
          letter-spacing: -0.05em;
          color: #f8fafc;
          text-shadow: 0 0 40px rgba(168, 85, 247, 0.22);
        }
        .hero-title .accent {
          background: linear-gradient(90deg, #c084fc, #a78bfa);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          text-shadow: 0 0 42px rgba(192, 132, 252, 0.55);
        }
        .particles-holder {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          z-index: 1;
          pointer-events: none;
        }
        .hero-scroll-hint {
          margin-top: 16px;
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 10px 16px;
          border-radius: 999px;
          color: rgba(226, 232, 240, 0.92);
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.08);
          backdrop-filter: blur(12px);
          font-size: 0.88rem;
          animation: floatHint 2.6s ease-in-out infinite;
        }
        .hero-scroll-hint::after {
          content: "↓";
          font-size: 1rem;
        }
        @keyframes floatHint {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(6px); }
        }
        @media (max-width: 940px) {
          .hero-features { grid-template-columns: 1fr; }
          .hero-inner { padding-top: 88px; }
        }
        </style>
        <div class="hero-screen">
          <div class="particles-holder" id="tsparticles"></div>
          <div class="hero-section">
            <div class="hero-inner">
              <div class="hero-kicker">Semantic Video Discovery</div>
              <div class="hero-logo">__LOGO_HTML_HERE__</div>
              <h1 class="hero-title">Search in <span class="accent">Video</span></h1>
              <p class="hero-copy">
                Upload a video, prepare it for search, and move through transcript, emotion, and summary workflows with a cleaner cinematic interface.
              </p>
              <div class="hero-features">
                <div class="hero-feature">
                  <p class="hero-feature-title">Instant clip search</p>
                  <p class="hero-feature-copy">Jump to the exact spoken moment with smooth result-to-player flow.</p>
                </div>
                <div class="hero-feature">
                  <p class="hero-feature-title">Smart results</p>
                  <p class="hero-feature-copy">Blend keyword matching, transcript context, and emotion-aware lookup.</p>
                </div>
                <div class="hero-feature">
                  <p class="hero-feature-title">Beautiful workflow</p>
                  <p class="hero-feature-copy">Glassmorphism cards, better spacing, and summary views that feel focused.</p>
                </div>
              </div>
              <a href="#main-cards" style="text-decoration: none; color: inherit;">
                <div class="hero-scroll-hint">Scroll to workspace</div>
              </a>
            </div>
          </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/tsparticles@3.8.1/tsparticles.min.js"></script>
        <script>
          const config = {
            fpsLimit: 60,
            particles: {
              number: { value: 220, density: { enable: true, area: 1300 } },
              color: { value: ["#ffffff", "#c084fc", "#7c3aed", "#86efac"] },
              shape: { type: "circle" },
              opacity: { value: { min: 0.25, max: 0.95 }, animation: { enable: true, speed: 1.3, minimumValue: 0.25 } },
              size: { value: { min: 1.8, max: 8 }, random: true },
              links: { enable: true, distance: 180, color: "#8b5cf6", opacity: 0.22, width: 1.2 },
              move: { enable: true, speed: 1.5, direction: "none", random: true, straight: false, outModes: { default: "bounce" }, trail: { enable: true, length: 20, fillColor: "#03030a" } }
            },
            interactivity: {
              detectsOn: "window",
              events: {
                onHover: { enable: true, mode: "bubble" },
                onClick: { enable: true, mode: "push" },
                resize: true
              },
              modes: {
                grab: { distance: 220, links: { opacity: 0.35 } },
                push: { quantity: 6 },
                bubble: { distance: 200, size: 10, duration: 2, opacity: 0.95 }
              }
            },
            detectRetina: true,
            background: { color: "transparent" }
          };
          const loadParticles = () => {
            if (window.tsParticles && window.tsParticles.load) {
              window.tsParticles.load("tsparticles", config);
            } else {
              setTimeout(loadParticles, 100);
            }
          };
          loadParticles();
        </script>
        """

    hero_html = hero_html.replace("__LOGO_HTML_HERE__", logo_html)
    st.components.v1.html(hero_html, height=980, scrolling=False)

    # Initialize upload state and cache loading flag
    if "uploaded_video_path" not in st.session_state:
        st.session_state.uploaded_video_path = DEFAULT_VIDEO_PATH
    if "segments" not in st.session_state:
        st.session_state.segments = []
        st.session_state.embeddings = None
        st.session_state.model = None
    if "cache_loaded" not in st.session_state:
        st.session_state.cache_loaded = False
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None

    # Restore prepared cache when possible (only once per session)
    if not st.session_state.cache_loaded and not st.session_state.segments and os.path.isfile(CACHE_FILE):
        cached_data = load_cache()
        if cached_data and os.path.isfile(cached_data.get("video_path", "")):
            st.session_state.uploaded_video_path = cached_data["video_path"]
            st.session_state.segments = cached_data["segments"]
            st.session_state.embeddings = cached_data["embeddings"]
            st.session_state.model = load_cached_model(cached_data.get("backend_name"))
            st.session_state.cache_loaded = True
            # st.success("Loaded prepared video from cache.")
            pass
        else:
            st.session_state.cache_loaded = True

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "Keyword"
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "selected_start" not in st.session_state:
        st.session_state.selected_start = 0.0
    if "autoplay_segment" not in st.session_state:
        st.session_state.autoplay_segment = False
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "show_summary" not in st.session_state:
        st.session_state.show_summary = False
    if "selected_result" not in st.session_state:
        st.session_state.selected_result = None
    if "scroll_to_video" not in st.session_state:
        st.session_state.scroll_to_video = False
    if "generated_summary" not in st.session_state:
        st.session_state.generated_summary = ""
    particles_html = build_background_particles_html()
    page_shell_html = """
        <style>
        #MainMenu, footer, header { visibility: hidden !important; }
        [data-testid="stSidebar"] { display: none !important; }
        .stApp, .main, section.main, .block-container {
          background:
            radial-gradient(circle at top, rgba(96, 165, 250, 0.07), transparent 18%),
            radial-gradient(circle at 82% 10%, rgba(168, 85, 247, 0.09), transparent 20%),
            #03030a !important;
          padding: 0 !important;
          margin: 0 auto !important;
        }
        .block-container { max-width: 100% !important; min-width: 100% !important; }
        .css-18e3th9, .css-1d391kg, .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
        .background-particles {
          position: fixed;
          inset: 0;
          z-index: 0;
          pointer-events: none;
          overflow: hidden;
        }
        /* Glowing Cinematic Border */
        .stApp::after {
          content: "";
          position: fixed;
          inset: 0;
          z-index: 999999;
          pointer-events: none;
          border: 1px solid rgba(0, 212, 255, 0.15);
          box-shadow: 
            inset 0 0 50px rgba(0, 212, 255, 0.2),
            inset 0 0 120px rgba(0, 212, 255, 0.05);
        }
        .bg-particle {
          position: absolute;
          top: -8vh;
          display: block;
          border-radius: 999px;
          background:
            radial-gradient(circle at 30% 30%, rgba(255,255,255,0.95), rgba(255,255,255,0.12) 58%, transparent 72%),
            linear-gradient(135deg, rgba(125, 211, 252, 0.85), rgba(192, 132, 252, 0.55));
          opacity: var(--particle-opacity);
          filter: blur(var(--particle-blur));
          box-shadow:
            0 0 14px rgba(125, 211, 252, 0.22),
            0 0 20px rgba(192, 132, 252, 0.16);
          animation: particleFall var(--particle-duration) linear infinite;
          animation-delay: calc(var(--particle-delay) * -1);
        }
        @keyframes particleFall {
          0% {
            transform: translate3d(0, -10vh, 0) scale(0.9);
            opacity: 0;
          }
          10% {
            opacity: var(--particle-opacity);
          }
          50% {
            transform: translate3d(22px, 48vh, 0) scale(1.08);
          }
          100% {
            transform: translate3d(-18px, 115vh, 0) scale(0.92);
            opacity: 0;
          }
        }
        .stButton > button { width: 100% !important; }
        .cards-wrapper { max-width: 1020px; width: min(100%, 1020px); margin: 0 auto; padding: 34px 20px 0; }
        div[data-testid="stHorizontalBlock"] {
          align-items: stretch !important;
          gap: 1.5rem !important;
          max-width: 1080px;
          width: min(100%, 1080px);
          margin: 0 auto 24px !important;
          padding: 0 10px;
          box-sizing: border-box;
        }
        div[data-testid="column"] {
          display: flex;
          align-self: stretch;
        }
        div[data-testid="column"] > div {
          width: 100%;
          display: flex;
          flex-direction: column;
        }
        .workspace-intro {
          max-width: 820px;
          margin: 0 auto 34px;
          text-align: center;
        }
        .workspace-kicker {
          display: inline-block;
          margin-bottom: 16px;
          padding: 10px 16px;
          border-radius: 999px;
          background: linear-gradient(135deg, rgba(125, 211, 252, 0.14), rgba(192, 132, 252, 0.12));
          border: 1px solid rgba(255, 255, 255, 0.08);
          color: #bae6fd;
          font-size: 0.82rem;
          font-weight: 800;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .workspace-title {
          margin: 0 0 10px;
          color: #f8fafc;
          font-size: clamp(2.3rem, 3.8vw, 3.8rem);
          line-height: 0.98;
          letter-spacing: -0.05em;
          font-weight: 950;
        }
        .workspace-copy {
          margin: 0 auto;
          max-width: 800px;
          color: #b7c4d8;
          font-size: 1.02rem;
          line-height: 1.9;
        }
        div[data-testid="column"] > div {
          width: 100%;
          display: flex;
          flex-direction: column;
          flex-grow: 1;
        }
        /* Target Streamlit's native bordered container */
        div[data-testid="column"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
          position: relative;
          display: flex;
          flex-direction: column;
          gap: 0;
          flex: 1 1 auto;
          height: 640px !important;
          min-height: 640px !important;
          max-height: 640px !important;
          position: relative !important;
          display: flex !important;
          flex-direction: column !important;
          padding: 30px 24px 30px !important;
          border-radius: 30px !important;
          border: 1px solid rgba(255,255,255,0.1) !important;
          background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.05)),
            linear-gradient(145deg, rgba(13, 20, 38, 0.9), rgba(7, 10, 22, 0.96)) !important;
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.12),
            0 30px 84px rgba(0, 0, 0, 0.4) !important;
          backdrop-filter: blur(22px);
          -webkit-backdrop-filter: blur(22px);
          opacity: 0;
          transform: translateY(22px);
          animation: fadeUp 0.9s cubic-bezier(0.22, 1, 0.36, 1) forwards;
          transition: transform 0.42s cubic-bezier(0.22, 1, 0.36, 1), filter 0.34s ease, border-color 0.36s ease, box-shadow 0.36s ease;
          text-align: left;
          overflow: hidden !important; /* Contain widgets */
        }
        
        /* Remove Streamlit's internal default border styling */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
           border: none !important;
           padding: 0 !important;
           background: transparent !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]::before {
          content: "";
          position: absolute;
          inset: 0;
          background:
            linear-gradient(180deg, rgba(255,255,255,0.08), transparent 18%),
            radial-gradient(circle at top right, rgba(125, 211, 252, 0.1), transparent 26%),
            radial-gradient(circle at bottom left, rgba(192, 132, 252, 0.08), transparent 28%);
          pointer-events: none;
          border-radius: 30px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]::after {
          content: "";
          position: absolute;
          left: 22px;
          right: 22px;
          top: 0;
          height: 1px;
          background: linear-gradient(90deg, rgba(125,211,252,0), rgba(125,211,252,0.46), rgba(192,132,252,0.46), rgba(192,132,252,0));
          pointer-events: none;
        }
        
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
          transform: translateY(-8px) !important;
          filter: brightness(1.045);
          border-color: rgba(192, 132, 252, 0.24) !important;
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 38px 110px rgba(76, 29, 149, 0.22) !important;
        }

        /* Delay animations for child elements */
        [data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlockBorderWrapper"] { animation-delay: 0.12s; }
        [data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] { animation-delay: 0.2s; }

        .card-header {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 0;
          min-height: 160px;
          margin-bottom: 20px;
          justify-content: flex-start;
        }
        .card-content {
          display: flex;
          flex-direction: column;
          gap: 18px;
          flex: 1 1 auto;
          width: 100%;
          min-height: 0;
        }
        .card-content-grow {
          justify-content: flex-start;
        }
        .video-card-anchor {
          position: absolute;
          width: 0;
          height: 0;
          overflow: hidden;
          opacity: 0;
          pointer-events: none;
        }
        .stButton button {
          transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
          border-radius: 16px !important;
          min-height: 2.9rem;
          background: linear-gradient(135deg, rgba(125, 211, 252, 0.94), rgba(192, 132, 252, 0.94)) !important;
          color: #08111f !important;
          font-weight: 800 !important;
          border: 0 !important;
          box-shadow: 0 18px 45px rgba(76, 29, 149, 0.24);
        }
        .stButton button:hover { transform: translateY(-2px); box-shadow: 0 24px 56px rgba(96, 165, 250, 0.22); }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stFileUploader section {
          border-radius: 18px !important;
          background: rgba(0, 0, 0, 0.6) !important;
          border: 1px solid rgba(255,255,255,0.12) !important;
          color: #f8fafc !important;
          box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        [data-testid="stFileUploader"] button {
          background: linear-gradient(135deg, rgba(30, 41, 59, 1), rgba(15, 23, 42, 1)) !important;
          color: #f1f5f9 !important;
          border: 1px solid rgba(255,255,255,0.1) !important;
          border-radius: 12px !important;
        }
        label p {
          color: #94a3b8 !important;
          font-weight: 700 !important;
          text-transform: uppercase;
          font-size: 0.75rem !important;
          letter-spacing: 0.05em;
        }
        .card-label {
          display: inline-flex;
          align-items: center;
          padding: 8px 14px;
          margin-bottom: 18px;
          border-radius: 999px;
          background: linear-gradient(135deg, rgba(125, 211, 252, 0.14), rgba(192, 132, 252, 0.18));
          color: #d8efff;
          font-size: 0.81rem;
          font-weight: 800;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          border: 1px solid rgba(255,255,255,0.08);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .card-title {
          margin: 0 0 14px;
          font-size: 1.78rem;
          font-weight: 900;
          color: #ffffff;
          line-height: 1.02;
          letter-spacing: -0.05em;
          max-width: 100%;
          text-wrap: pretty;
        }
        .card-copy {
          margin: 0;
          color: #c9d3e6;
          line-height: 1.84;
          font-size: 1rem;
          max-width: 48ch;
          font-style: italic;
        }
        .card-subtle {
          display: inline-flex;
          align-items: center;
          margin: 0 0 18px;
          padding: 11px 15px;
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
          border: 1px solid rgba(255,255,255,0.08);
          color: #d9e5f6;
          font-size: 0.92rem;
          line-height: 1.5;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
          font-style: italic;
        }
        .card-subtle strong {
          color: #ffffff;
          margin-left: 6px;
        }
        .card-panel .stButton {
          margin-top: auto;
          padding-top: 8px;
        }
        .card-panel [data-testid="stFileUploader"] {
          margin-bottom: 18px;
        }
        .card-panel [data-testid="stTextInput"],
        .card-panel [data-testid="stSelectbox"] {
          margin-bottom: 12px;
        }
        .card-panel [data-testid="stAlert"] {
          margin-top: 14px;
          margin-bottom: 0;
        }
        .results-list { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; width: 100%; }
        .results-shell {
          max-width: 1240px;
          margin: 28px auto 0;
          padding: 0 72px 46px;
        }
        .results-item { margin-bottom: 0; padding: 18px 18px 16px; border-radius: 22px; background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025)); border: 1px solid rgba(255,255,255,0.06); transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease; display: flex; flex-direction: column; height: 100%; min-height: 200px; }
        .results-item:hover { transform: translateY(-2px); border-color: rgba(125, 211, 252, 0.18); box-shadow: 0 18px 40px rgba(96,165,250,0.10); }
        .results-meta { font-size: 0.92rem; color: #a5b4fc; margin-bottom: 10px; font-weight: 700; letter-spacing: 0.01em; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
        .results-text { margin: 0; color: #e2e8f0; line-height: 1.78; font-size: 0.98rem; text-indent: 0.2rem; flex-grow: 1; display: flex; align-items: center; }
        .results-button-wrapper { display: flex; gap: 10px; margin-top: auto; }
        .selected-badge { display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px; background: rgba(139,92,246,0.14); color: #e0def8; font-weight: 700; margin-bottom: 16px; }
        .card-button { width: 100%; padding: 14px 22px; border-radius: 999px; border: none; background: linear-gradient(135deg, rgba(139,92,246,0.92), rgba(129,140,248,0.92)); color: #ffffff; font-weight: 800; cursor: pointer; transition: transform 0.25s ease, box-shadow 0.25s ease; box-shadow: 0 0 0 rgba(139,92,246,0.0); }
        .card-button:hover { transform: translateY(-2px); box-shadow: 0 25px 60px rgba(139,92,246,0.18); }
        .scroll-anchor {
          position: relative;
          top: -18px;
          visibility: hidden;
        }
        .content-reveal {
          animation: revealSection 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
        }
        @keyframes revealSection {
          from { opacity: 0; transform: translateY(28px); }
          to { opacity: 1; transform: translateY(0); }
        }
        ::-webkit-scrollbar { width: 12px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
        ::-webkit-scrollbar-thumb {
          border-radius: 999px;
          background: linear-gradient(180deg, rgba(125, 211, 252, 0.65), rgba(192, 132, 252, 0.65));
          border: 3px solid rgba(3,3,10,0.94);
        }
        @media (max-width: 940px) { .cards-row { grid-template-columns: 1fr; } }
        @media (max-width: 900px) {
          .cards-wrapper, .results-shell { padding-left: 20px; padding-right: 20px; }
          div[data-testid="stHorizontalBlock"] {
            padding-left: 20px;
            padding-right: 20px;
            gap: 1rem !important;
          }
          .workspace-intro { margin-bottom: 18px; }
          div[data-testid="column"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
            height: auto !important;
            min-height: 480px !important;
            max-height: none !important;
          }
        }
        @media (max-width: 640px) {
          div[data-testid="column"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
            height: auto !important;
            min-height: auto !important;
            max-height: none !important;
          }
          .card-panel { padding: 22px 18px 24px; }
          div[data-testid="stHorizontalBlock"] {
            padding-left: 14px;
            padding-right: 14px;
          }
          .card-header { min-height: auto; }
          .hero-title { font-size: 3.25rem; }
          .card-title { font-size: 1.4rem; }
          .workspace-title { font-size: 2rem; }
        }
        </style>
        __PARTICLES_HTML__
        <div id="main-cards" class="scroll-anchor"></div>
        <div class="cards-wrapper content-reveal">
          <div class="workspace-intro">
            <div class="workspace-kicker">Workspace</div>
            <h2 class="workspace-title">A smoother way to search through long videos</h2>
            <p class="workspace-copy">The workflow below is tuned for clarity: prepare once, search fast, preview instantly, and open focused summaries without breaking the flow.</p>
          </div>
        </div>
        """
    st.markdown(page_shell_html.replace("__PARTICLES_HTML__", particles_html), unsafe_allow_html=True)



    video_path = st.session_state.uploaded_video_path
    search_query = st.session_state.search_query
    search_mode = st.session_state.search_mode

    row_one = st.columns(2, gap="large")
    row_two = st.columns(2, gap="large")

    with row_one[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">'
                '<span class="card-label">01</span>'
                '<h3 class="card-title">Upload Video</h3>'
                '<p class="card-copy">Upload or drop a video to use for search.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "mkv", "avi"], key="video_uploader")
            if uploaded_file is not None:
                # Only process if this is a NEW file (not the same file from a previous rerun)
                if st.session_state.last_uploaded_filename != uploaded_file.name:
                    ensure_output_dir()
                    saved_path = os.path.join(OUTPUT_DIR, "uploaded_video.mp4")
                    with open(saved_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.uploaded_video_path = saved_path
                    st.session_state.segments = []
                    st.session_state.embeddings = None
                    st.session_state.model = None
                    st.session_state.search_results = []
                    st.session_state.selected_start = 0.0
                    st.session_state.selected_result = None
                    st.session_state.cache_loaded = False  # Reset cache flag for new video
                    st.session_state.last_uploaded_filename = uploaded_file.name  # Track this file
                    clear_cache()  # Clear old cache for new video
                    # st.success("Video uploaded and saved. Ready to prepare for search.")
                    pass
                    video_path = saved_path
                else:
                    # Same file, use the saved path
                    video_path = st.session_state.uploaded_video_path
            else:
                # No file uploaded, use the current path
                video_path = st.session_state.uploaded_video_path

            if os.path.isfile(video_path):
                st.markdown(f"<div class='card-subtle'>Current source:<strong>{os.path.basename(video_path)}</strong></div>", unsafe_allow_html=True)
            else:
                st.warning("Please upload a valid video file to start.")

            if st.button("Prepare video", key="prepare_video"):
                if not os.path.isfile(video_path):
                    st.warning("Upload a video first before preparing.")
                else:
                    try:
                        st.info("Extracting audio and transcribing your video. This may take a while...")
                        audio_path = os.path.join(OUTPUT_DIR, AUDIO_FILENAME)
                        extract_audio(video_path, audio_path)
                        whisper_model = load_whisper_model()
                        st.session_state.segments = transcribe_audio(audio_path, model=whisper_model)
                        
                        if not st.session_state.segments:
                            st.error("No speech detected in the video. Please check if the video has audio.")
                        else:
                            # Detect emotions for all segments
                            st.session_state.segments = detect_all_segment_emotions(st.session_state.segments)
                            
                            try:
                                st.session_state.embeddings, st.session_state.model = create_embeddings(st.session_state.segments)
                                backend_name = getattr(st.session_state.model, "backend_name", "sentence-transformers")
                                if backend_name != "sentence-transformers":
                                    st.info(
                                        "Video prepared using fallback text matching. Search and emotion matching will work, "
                                        "but results may be less accurate than the transformer model."
                                    )
                            except Exception as exc:
                                st.session_state.embeddings = None
                                st.session_state.model = None
                                st.warning(
                                    "Video prepared with transcript only. Semantic and emotion search are unavailable right now. "
                                    f"Reason: {exc}"
                                )
                            st.session_state.search_results = []
                            st.session_state.selected_start = 0.0
                            st.session_state.selected_result = None
                            st.session_state.cache_loaded = True  # Mark cache as loaded after prepare
                            # Save to persistent cache
                            save_cache(video_path, st.session_state.segments, st.session_state.embeddings, st.session_state.model)
                            # st.success(f"✅ Ready to search: {len(st.session_state.segments)} segments generated.")
                            pass
                            st.info("🎭 Emotion detection complete - emotions analyzed for all segments!")
                            st.rerun()  # Force rerun to ensure session state is properly synced
                    except Exception as e:
                        st.error(f"Error preparing video: {str(e)}")
                        st.session_state.segments = []
                        st.session_state.embeddings = None
                        st.session_state.model = None

    with row_one[1]:
        with st.container(border=True):
            st.markdown("<div id='video-card' class='video-card-anchor'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="card-header">'
                '<span class="card-label">02</span>'
                '<h3 class="card-title">Uploaded Video</h3>'
                '<p class="card-copy">Play the video you uploaded and preview selected search results here.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if os.path.isfile(video_path):
                render_video_player(
                    video_path, 
                    start_time=st.session_state.selected_start, 
                    height=300, 
                    autoplay=st.session_state.autoplay_segment
                )
                # Reset autoplay flag after rendering to video instance
                if st.session_state.autoplay_segment:
                    st.session_state.autoplay_segment = False
                
                if st.session_state.selected_start > 0:
                    st.markdown(f"<div class='selected-badge'>Playing segment at {st.session_state.selected_start:.2f}s</div>", unsafe_allow_html=True)
            else:
                st.warning("No uploaded video available yet.")

    with row_two[0]:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">'
                '<span class="card-label">03</span>'
                '<h3 class="card-title">Search in Video</h3>'
                '<p class="card-copy">Search by keyword or emotion. Results will show in the video preview card.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            search_query = st.text_input("Search query", value=search_query, key="search_query")
            search_mode = st.selectbox("Search mode", ["Keyword", "Emotion"], index=0, key="search_mode")
            if st.button("Search", key="search_button"):
                if not os.path.isfile(video_path):
                    st.warning("Upload and prepare a video first before searching.")
                elif not st.session_state.segments:
                    st.warning("Prepare the video first before searching.")
                else:
                    query = search_query.strip()
                    if not query:
                        st.warning("Enter text or emotion to search.")
                    else:
                        if search_mode == "Keyword":
                            results = find_segments(query, st.session_state.segments)
                        else:
                            # Use the emotion classifier for emotion detection
                            classifier = load_emotion_classifier()
                            if classifier is None:
                                st.warning(
                                    "⏳ Emotion classifier is loading... This may take a moment on first run. Please try again in a few seconds."
                                )
                                results = []
                            else:
                                # Find segments using the new classifier-based search
                                results = find_segments_by_emotion_classifier(query, st.session_state.segments)
                                if results:
                                    st.info(f"🎭 Searching for emotion: {query.upper()}")
                                else:
                                    # Fallback to keyword search if no emotion matches
                                    results = find_segments(query, st.session_state.segments)
                                    if results:
                                        st.info(f"No emotion match found - showing keyword results for: {query.upper()}")
                        st.session_state.search_results = results
                        st.session_state.selected_start = 0.0
                        st.session_state.selected_result = None
                        if results:
                            st.session_state.show_results = True
                            st.rerun()
                        else:
                            st.warning("No matching segments found.")

            if st.session_state.search_results:
                st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
                if st.button("🔍 View Latest Results", use_container_width=True):
                    st.session_state.show_results = True
                    st.rerun()

    with row_two[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header">'
                '<span class="card-label">04</span>'
                '<h3 class="card-title">Generate Summary</h3>'
                '<p class="card-copy">Create a comprehensive summary of the entire video transcript directly below.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if not os.path.isfile(video_path):
                st.warning("Upload a valid video first to generate a summary.")
            elif not st.session_state.segments:
                st.warning("Prepare the video first to enable summary generation.")
            else:
                if st.button("Generate Summary", key="generate_summary_btn", use_container_width=True):
                    with st.spinner("Generating your transcript summary..."):
                        st.session_state.generated_summary = generate_comprehensive_summary(
                            st.session_state.segments,
                            target_length=900,
                        ) or "Summary generation returned no text."
                        st.success("Summary generated successfully.")
                        st.session_state.show_summary = True
                        st.session_state.show_results = False # Close other dialog
                        st.rerun()

    # Mutually exclusive dialog rendering at the end of the script
    if st.session_state.show_summary and st.session_state.generated_summary:
        show_summary_dialog(st.session_state.generated_summary)
    elif st.session_state.show_results and st.session_state.search_results:
        show_results_dialog(st.session_state.search_results)

    if st.session_state.scroll_to_video:
        st.markdown(
            "<script>const e=document.getElementById('video-card'); if(e){e.scrollIntoView({behavior:'smooth', block:'center'});}</script>",
            unsafe_allow_html=True,
        )
        st.session_state.scroll_to_video = False

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)


def print_dependencies():
    print("Dependencies required:")
    print(" - openai-whisper")
    print(" - sentence-transformers")
    print(" - faiss-cpu")
    print(" - streamlit")
    print(" - ffmpeg installed on your system")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "cli":
            run_cli()
            return
        if mode == "streamlit":
            if st is None:
                raise ImportError("streamlit is not installed. Install streamlit to use the UI.")
            streamlit_app()
            return
        if mode in {"help", "-h", "--help"}:
            print("Usage: python src/main.py [cli|streamlit]")
            print("Or run the UI with: streamlit run src/main.py")
            return

    if st is not None:
        streamlit_app()
    else:
        print("Run the app with Streamlit:")
        print("  streamlit run src/main.py")
        print("Or run the CLI with:")
        print("  python src/main.py cli")
        print_dependencies()


if __name__ == "__main__":
    main()
