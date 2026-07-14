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


@st.cache_resource(show_spinner="Loading page index...")
def load_page_index():
    """Insaan ne khud har page ka topic likha hai (Excel se banaya gaya) —
    yeh sabse zyada bharosemand tareeqa hai sahi page dhoondne ka, kyunke
    AI ke guess/OCR noise par depend nahi karta."""
    path = "data/page_index.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# Sirf halki/tez cheezein app khulte hi load karte hain (JSON files, API
# client). Bhari AI models (embedding, voice recognition) ko JAAN-BOOJH
# KAR yahan load NAHI karte — warna app tab tak kuch nahi dikhati jab tak
# yeh dono load na ho jayen (jo Streamlit Cloud ke free/limited resources
# par bohat der lag sakti hai, kabhi kabhi hang bhi ho sakta hai). Inhein
# neeche 'get_embedder()' aur 'get_stt_model()' se sirf USI waqt load
# karte hain jab pehli baar zaroorat pade — is se interface turant khulta
# hai, aur AI model sirf tab load hota hai jab wakai use ho raha ho.
chunk_data, index = load_chunks_and_index()
page_index = load_page_index()
client = load_groq_client()


def get_embedder():
    return load_embedder()


def get_stt_model():
    return load_whisper_model()

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
    chunk 'text' aur 'page' dono rakhta hai.

    Scoring 3 tiers mein: (1) EXACT PHRASE match (jaisay 'boundary wall'
    poora sath likha ho) sabse zyada bharosemand hai — sabse upar. (2) Kitne
    ALAG alag query words maujood hain (distinct hits). (3) Total frequency
    sirf tie-breaker. Isse sirf ek generic lafz (jaisay akela 'block') kisi
    galat page ko upar nahi le ja sakta agar wahan poora phrase maujood
    nahi hai."""
    words = list(set(w for w in re.findall(r"[A-Za-z]+", query.lower()) if len(w) > 2))
    if not words:
        return []

    ordered_words = re.findall(r"[A-Za-z]+", query.lower())
    phrases = [f"{ordered_words[i]} {ordered_words[i+1]}" for i in range(len(ordered_words) - 1)]

    scored = []
    for i, c in enumerate(chunk_data):
        if is_low_value_chunk(c["text"]):
            continue
        chunk_lower = c["text"].lower()
        distinct_hits = sum(1 for w in words if w in chunk_lower)
        if distinct_hits == 0:
            continue
        phrase_hits = sum(1 for p in phrases if p in chunk_lower)
        total_count = sum(chunk_lower.count(w) for w in words)
        scored.append((phrase_hits, distinct_hits, total_count, i))
    scored.sort(reverse=True)
    return [i for *_, i in scored[:top_n]]


def find_manual_page_match(query):
    """Insaan ke likhe hue 'Page Index' (Excel se) mein query ke alfaz
    dhoondta hai. Yeh topic descriptions bohat chote aur saaf hain (OCR
    noise nahi), is liye yahan match milna sabse zyada bharosemand hai.

    'Y' (asal drawing/measurements wala page) ko hamesha priority dete hain
    'N' (sirf photo) par — kyunke sawal poochne wale ko aksar specs/details
    chahiye hoti hain, tasveer nahi. Agar sirf photo match mila, to uska
    juda hua detail page (aik page aagay, usi topic ka) bhi dhoondte hain."""
    if not page_index:
        return None

    words = list(set(w for w in re.findall(r"[A-Za-z]+", query.lower()) if len(w) > 2))
    if not words:
        return None

    ordered_words = re.findall(r"[A-Za-z]+", query.lower())
    phrases = [f"{ordered_words[i]} {ordered_words[i+1]}" for i in range(len(ordered_words) - 1)]

    scored = []
    for entry in page_index:
        topic_lower = entry["topic"].lower()
        distinct_hits = sum(1 for w in words if w in topic_lower)
        if distinct_hits == 0:
            continue
        phrase_hits = sum(1 for p in phrases if p in topic_lower)
        scored.append((phrase_hits, distinct_hits, entry))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if not scored:
        return None

    # Y (drawing/detail) wale matches ko upar rakhte hain
    y_matches = [e for _, _, e in scored if e["has_drawing"]]
    n_matches = [e for _, _, e in scored if not e["has_drawing"]]

    if y_matches:
        return y_matches[0]

    if n_matches:
        # Sirf photo mila — uska juda hua detail (Y) page dhoondte hain
        # (pattern: pic page ke theek baad wala page usi topic ka detail hai)
        pic_entry = n_matches[0]
        next_page = pic_entry["page"] + 1
        for entry in page_index:
            if entry["page"] == next_page and entry["has_drawing"]:
                return entry
        return pic_entry

    return None


def get_relevant_chunks(query, k=6):
    """Ab yeh sirf text nahi, poora chunk dict ({"text", "page"}) return
    karta hai taake page number bhi pata rahe."""
    embedder = get_embedder()
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

    # STEP 1: Pehle manual (insaan ke likhe hue) Page Index mein dhoondein —
    # yeh sabse reliable hai. RAW sawal (jaisa likha/bola gaya, bina AI se
    # 'clean' karwaye) se PEHLE try karte hain — kyunke AI cleaning step
    # kabhi kabhi lafz badal deta hai (jaise 'walkway' ko kisi aur lafz mein
    # rephrase kar dena), jo hamari saaf Excel list se match nahi karta.
    # Sirf agar raw text se match na mile (jaise Urdu/Roman Urdu awaaz),
    # tab translated/cleaned version try karte hain.
    manual_match = find_manual_page_match(user_question)
    if not manual_match and retrieval_query:
        manual_match = find_manual_page_match(retrieval_query)

    if manual_match:
        matched_page = manual_match["page"]
        matched_chunks = [c for c in chunk_data if c["page"] == matched_page]
        pages_used = [matched_page]
    else:
        # STEP 2: Manual index mein match nahi mila — purani AI-based
        # hybrid (keyword + semantic) search par fallback karte hain.
        matched_chunks = get_relevant_chunks(query_for_search)
        pages_used = []
        for c in matched_chunks:
            if c["page"] not in pages_used:
                pages_used.append(c["page"])

    context = "\n\n".join(c["text"] for c in matched_chunks)

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
    """Voice/text sawal ko English mein translate karta hai taake search ho
    sake. Jaan-boojh kar sirf 'literal translation' maangte hain (summarize
    ya 'core topic nikalna' nahi) — kyunke translate+summarize dono ek sath
    karne se chota/tez model kabhi kabhi topic hi badal deta hai (jaise
    'toilet' ko 'pantry' samajh lena). Filler words hata dena hamari apni
    matching logic khud sambhal leti hai, LLM se yeh na karwana zyada
    reliable hai."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "Translate the user's message to English, word for word, as literally and "
                    "accurately as possible. If it is already in English, repeat it unchanged. "
                    "Do NOT summarize, paraphrase, explain, or change the meaning. "
                    "Reply with ONLY the direct English translation, nothing else."
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
    stt_model = get_stt_model()
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
            pages_used = msg["pages"]

            # Sirf TOP 2 sabse relevant pages mein se image dhoondte hain
            # (poori list mein nahi) — taake dono cheezein balance hon:
            # (1) agar sabse top page ki drawing na ho to bhi 2nd-most
            # relevant page try ho jaye, (2) lekin list ke aakhri (kam
            # relevant) pages ki galat/na-related image kabhi na dikhe.
            shown_image_page = None
            for p in pages_used[:2]:
                candidate_path = f"data/pages/page_{p}.jpg"
                if os.path.exists(candidate_path):
                    shown_image_page = p
                    break

            if shown_image_page:
                st.caption(f"📄 Drawing reference: Page {shown_image_page}")
                st.image(f"data/pages/page_{shown_image_page}.jpg", use_container_width=True)

            st.caption(f"Source page(s): {', '.join(str(p) for p in pages_used[:4])}")

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
