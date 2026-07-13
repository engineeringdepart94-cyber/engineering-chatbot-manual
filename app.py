import time
import re
import os
import json
import tempfile
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq, RateLimitError
from gtts import gTTS
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Engineering Standard AI Chatbot (with Pages)", page_icon="📄", layout="centered")

# ---------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource(show_spinner="Loading PDF data...")
def load_chunks_and_index():
    with open("data/chunks_with_pages.json", "r", encoding="utf-8") as f:
        chunk_data = json.load(f)  # list of {"text": ..., "page": ...}
    index = faiss.read_index("data/faiss_index.bin")
    return chunk_data, index


@st.cache_resource(show_spinner="Connecting to AI...")
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error("GROQ_API_KEY not found. Please add it in Streamlit Cloud Settings > Secrets.")
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner="Loading voice recognition model...")
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")


embedder = load_embedder()
chunk_data, index = load_chunks_and_index()
client = load_groq_client()
stt_model = load_whisper_model()

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def is_low_value_chunk(text):
    """Kuch pages (jaise Table of Contents, index) sirf headings ki list hoti
    hain — inmein keywords bohat zyada density se milte hain (e.g. 'Boundary
    Wall (Block)', 'Boundary Wall (Brick)' lagataar likhe hote hain) lekin
    asal detail/specs is page par nahi hoti. Aise chunks ko hum search se
    bahar rakhte hain taake yeh galti se 'sabse relevant page' na ban jayen."""
    lowered = text.lower()
    skip_markers = ["table of contents", "list of illustrations", "list of figures"]
    return any(marker in lowered for marker in skip_markers)


def keyword_search_chunks(query, top_n=6):
    """Exact keyword matching, jaisay pehle wali app mein — sirf ab har
    chunk 'text' aur 'page' dono rakhta hai."""
    words = list(set(w for w in re.findall(r"[A-Za-z]+", query.lower()) if len(w) > 2))
    if not words:
        return []
    scored = []
    for i, c in enumerate(chunk_data):
        if is_low_value_chunk(c["text"]):
            continue
        chunk_lower = c["text"].lower()
        distinct_hits = sum(1 for w in words if w in chunk_lower)
        if distinct_hits == 0:
            continue
        total_count = sum(chunk_lower.count(w) for w in words)
        scored.append((distinct_hits, total_count, i))
    scored.sort(reverse=True)
    return [i for _, _, i in scored[:top_n]]


def get_relevant_chunks(query, k=6):
    """Ab yeh sirf text nahi, poora chunk dict ({"text", "page"}) return
    karta hai taake page number bhi pata rahe."""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    semantic_indices = [i for i in indices[0] if not is_low_value_chunk(chunk_data[i]["text"])]

    keyword_indices = keyword_search_chunks(query, top_n=6)

    combined_indices = []
    for a, b in zip(keyword_indices, semantic_indices):
        for i in (a, b):
            if i not in combined_indices:
                combined_indices.append(i)
    for i in list(keyword_indices) + list(semantic_indices):
        if i not in combined_indices:
            combined_indices.append(i)

    top_indices = combined_indices[:7]
    return [chunk_data[i] for i in top_indices]


def ask_ai(user_question, retrieval_query=None):
    query_for_search = retrieval_query if retrieval_query else user_question
    matched_chunks = get_relevant_chunks(query_for_search)
    context = "\n\n".join(c["text"] for c in matched_chunks)

    # Sab pages jahan se context liya gaya, order preserve karte hue
    # (pehla page = sabse zyada relevant match)
    pages_used = []
    for c in matched_chunks:
        if c["page"] not in pages_used:
            pages_used.append(c["page"])

    system_prompt = (
        "Tum ek engineering assistant ho. Sirf diye gaye context se jawab do. "
        "Jis zaban mein user sawal karay (Urdu, Roman Urdu, ya English), usi zaban mein jawab do. "
        "Agar context mein jawab na mile to bolo 'Yeh information PDF mein maujood nahi'."
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nSawal: {user_question}"},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content, pages_used
        except RateLimitError as e:
            if attempt == 0:
                time.sleep(8)
                continue
            return (
                "⏳ Too many questions were asked in a short time, so the free AI quota "
                "has been used up for now. Please wait a moment and ask again.\n\n"
                f"(Technical detail: {e})"
            ), []
        except Exception as e:
            return f"⚠️ Something went wrong, please try again. ({e})", []


def get_search_query(user_question):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "Extract only the core technical/engineering topic being asked about, "
                    "in English, as a short search phrase (2-6 words). "
                    "Remove filler words like 'batao', 'tell me', 'what is', 'detail', 'kya hai', 'please'. "
                    "Reply with ONLY the search phrase, nothing else."
                )},
                {"role": "user", "content": user_question},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return user_question


def detect_response_lang(text):
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return "ur"
    return "en"


def generate_voice(text):
    lang = detect_response_lang(text)
    tts = gTTS(text=text, lang=lang, slow=False)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(path)
    return path


def transcribe_audio_bytes(audio_bytes):
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    segments, info = stt_model.transcribe(tmp_path, language="ur")
    text = " ".join([seg.text for seg in segments])
    return text.strip()


def handle_question(user_question, tag="", retrieval_query=None):
    st.session_state.messages.append({"role": "user", "content": f"{tag}{user_question}"})
    with st.spinner("Preparing your answer..."):
        answer, pages_used = ask_ai(user_question, retrieval_query=retrieval_query)
        audio_path = generate_voice(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "audio": audio_path,
        "pages": pages_used,
    })


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------

st.title("📄 Engineering Standard AI Chatbot")
st.caption("Type or speak your question — the source PDF page is shown alongside every answer")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")
        if msg["role"] == "assistant" and msg.get("pages"):
            top_page = msg["pages"][0]
            other_pages = msg["pages"][1:4]  # baaki pages sirf naam se, image nahi (clutter na ho)
            img_path = f"data/pages/page_{top_page}.jpg"
            st.caption(f"📄 Found on Page {top_page}")
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"⚠️ Page image not found at: {img_path} (check this file exists in the GitHub repo)")
            if other_pages:
                st.caption(f"Also mentioned on page(s): {', '.join(str(p) for p in other_pages)}")

st.divider()
st.write("🎤 Ask by voice:")
audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key="mic")

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if audio and audio.get("id") != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio.get("id")
    with st.spinner("Listening to your voice..."):
        transcribed = transcribe_audio_bytes(audio["bytes"])
    if transcribed:
        with st.spinner("Preparing a better answer..."):
            search_query = get_search_query(transcribed)
        handle_question(transcribed, tag="🎤 ", retrieval_query=search_query)
        st.rerun()
    else:
        st.warning("Sorry, I could not understand that. Please try again.")

user_text = st.chat_input("Type your question...")
if user_text:
    with st.spinner("Preparing a better answer..."):
        search_query = get_search_query(user_text)
    handle_question(user_text, retrieval_query=search_query)
    st.rerun()
