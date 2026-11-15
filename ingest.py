import os
import re
import sys
import psycopg2
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client()


MAX_FILE_MB = 5
MAX_PAGES = 100
MAX_CHUNKS = 300
BATCH_SIZE = 50


pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
if size_mb > MAX_FILE_MB:
    raise SystemExit(f"File is {size_mb:.1f} MB. The limit is {MAX_FILE_MB} MB.")

reader = PdfReader(pdf_path)
if len(reader.pages) > MAX_PAGES:
    raise SystemExit(f"PDF has {len(reader.pages)} pages. The limit is {MAX_PAGES}.")

full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

if not full_text.strip():
    raise SystemExit("No text found in the PDF. It may be scanned.")


def sentence_chunks(text, max_size=500):
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_size:
            chunks.append(current)
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current)
    return chunks

chunks = sentence_chunks(full_text)

if len(chunks) > MAX_CHUNKS:
    raise SystemExit(f"{len(chunks)} chunks is too many. The limit is {MAX_CHUNKS}.")


vectors = []
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch
    )
    vectors.extend(e.values for e in result.embeddings)


conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("TRUNCATE chunks RESTART IDENTITY")   

for chunk, vec in zip(chunks, vectors):
    cur.execute(
        "INSERT INTO chunks (content, embedding) VALUES (%s, %s::vector)",
        (chunk, str(vec))
    )

conn.commit()

cur.execute("SELECT COUNT(*) FROM chunks")
print("Rows in chunks table:", cur.fetchone()[0])

cur.close()
conn.close()