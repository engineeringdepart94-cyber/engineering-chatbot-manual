# 📄 Engineering Standard AI Chatbot — with Source Page Display

Yeh pehli app jaisi hi hai (type/voice se sawal poochna, Urdu/Roman Urdu/English,
AI voice jawab) — **farq sirf itna hai ke har jawab ke sath PDF ka wo page bhi
dikhta hai jahan se information li gayi hai.**

---

## Folder structure (jaisi honi chahiye)

```
engineering-chatbot-pages/
├── app.py
├── requirements.txt
├── packages.txt
├── .gitignore
├── README.md
├── .streamlit/
│   └── secrets.toml.example
└── data/
    ├── chunks_with_pages.json
    ├── faiss_index.bin
    └── pages/
        ├── page_1.jpg
        ├── page_2.jpg
        └── ... (PDF ke tamam pages)
```

⚠️ `data/chunks_with_pages.json`, `data/faiss_index.bin`, aur `data/pages/` folder
(sab page images ke sath) — yeh teeno cheezein **Colab se banani hain** (README
ke neechay tareeqa hai) aur is folder mein daalni hain.

---

## Step 1: Colab mein data banayein

Apni Colab notebook mein (jahan `pdf_path` pehle se available hai), yeh naye
cells chalayein:

```python
import fitz
import pytesseract
from PIL import Image
import io
import os

os.makedirs("pages_images", exist_ok=True)
doc = fitz.open(pdf_path)
OCR_LANG = 'eng'
page_texts = []

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    page_texts.append(text)
    img.convert("RGB").save(f"pages_images/page_{i+1}.jpg", "JPEG", quality=70)
    if (i + 1) % 5 == 0 or (i + 1) == len(doc):
        print(f"✅ {i+1}/{len(doc)} pages done")
```

```python
def chunk_page_text(text, page_num, chunk_size=800, overlap=100):
    result = []
    if len(text.strip()) == 0:
        return result
    start = 0
    while start < len(text):
        end = start + chunk_size
        result.append({"text": text[start:end], "page": page_num})
        start += chunk_size - overlap
    return result

all_chunks = []
for i, text in enumerate(page_texts):
    all_chunks.extend(chunk_page_text(text, page_num=i + 1))
```

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
texts_only = [c["text"] for c in all_chunks]
chunk_embeddings = embedder.encode(texts_only, show_progress_bar=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))
```

```python
import json

with open("chunks_with_pages.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False)

faiss.write_index(index, "faiss_index.bin")
!zip -r pages_images.zip pages_images

from google.colab import files
files.download("chunks_with_pages.json")
files.download("faiss_index.bin")
files.download("pages_images.zip")
```

`pages_images.zip` ko apne computer par **extract** kar lein — andar
`page_1.jpg`, `page_2.jpg` waghera milengi. Inhein `data/pages/` folder
mein daal dein (JSON aur bin file `data/` mein).

---

## Step 2: GitHub par upload karein

1. Naya repository banayein (jaise `engineering-chatbot-pages`) — Public rakhein
2. Yeh sab files/folders upload karein:
   - `app.py`, `requirements.txt`, `packages.txt`, `.gitignore`, `README.md`
   - `data/chunks_with_pages.json`
   - `data/faiss_index.bin`
   - `data/pages/` (poora folder, sari page images ke sath — folder ko seedha
     drag-and-drop karein upload screen par, GitHub folder structure preserve
     kar lega)
3. **Commit changes**

---

## Step 3: Streamlit Cloud par deploy karein

1. https://share.streamlit.io → **Create app** → **Deploy a public app from GitHub**
2. Repository: `engineering-chatbot-pages`, Branch: `main`, Main file: `app.py`
3. **Advanced settings → Secrets**:
   ```
   GROQ_API_KEY = "apni_asal_groq_key"
   ```
4. **Deploy**

2-5 minute mein public link mil jayega.

---

## Yeh app extra kya karti hai

Jab AI jawab deta hai, Python khud (AI se nahi) track karta hai ke jawab
kis PDF chunk(s) se aaya, aur un chunks ke page number nikal kar:
- Sabse zyada relevant page ki **poori image dikhati hai**
- Agar jawab kai pages se related ho, baaki pages sirf naam se mention
  karti hai (clutter na ho)

## Notes

- Agar PDF mein bohat zyada pages hain, `pages_images.zip` bhi bara ho sakta
  hai — GitHub par upload karne mein thora time lag sakta hai, normal hai
- Naya PDF process karna ho to Step 1 dobara chalayein aur data update kar dein
