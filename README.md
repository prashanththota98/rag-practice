# RAG Document Q&A (Gemini + pgvector)

A small Retrieval-Augmented Generation (RAG) project in Python. It reads a PDF, stores it as
searchable embeddings in Postgres (pgvector), and answers questions using only the retrieved text.

The included `sample.pdf` is a fictional annual report made for practice. All names and figures
in it are invented.

## How it works

**Ingest (run once per document):**
PDF -> extract text -> sentence-aware chunks -> Gemini embeddings -> stored in Postgres (pgvector)

**Ask (run for every question):**
Question -> embedding -> cosine-distance search for the 4 closest chunks -> prompt containing
those chunks -> Gemini answers using only that context -> answer + source chunks

## Tech stack

- Python 3.10
- pypdf for PDF text extraction
- Google Gemini API: `gemini-embedding-001` (embeddings) and `gemini-3.5-flash-lite` (answers)
- Postgres with the pgvector extension (hosted on Supabase)
- psycopg2 and python-dotenv

## Setup

1. Clone the repo and install the packages:
   ```
   git clone https://github.com/prashanththota98/rag-practice
   cd rag-practice
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Create a Postgres database that supports pgvector (for example a free Supabase project) and
   run `schema.sql` in its SQL editor.
3. Copy `.env.example` to `.env` and fill in:
   - `GEMINI_API_KEY`: a free key from Google AI Studio
   - `DATABASE_URL`: your Postgres connection string (no extra `?` options at the end)
4. Load a document:
   ```
   python ingest.py                 # uses sample.pdf
   python ingest.py yourfile.pdf    # uses your own text-based PDF
   ```
5. Ask questions:
   ```
   python ask.py
   ```

## Design choices

- **Chunking:** sentences are grouped into chunks of up to 500 characters, so no sentence is
  cut in half. I first tried fixed-size chunks with overlap, which sometimes cut words in the middle.
- **Embeddings:** each chunk is embedded once at ingest time and stored, so every question needs
  only one embedding call. Chunks are sent in batches of 50.
- **Retrieval:** the top 4 chunks by cosine distance (`<=>` in pgvector). No vector index is used,
  because the data is tiny and the standard pgvector indexes only support up to 2000 dimensions
  (these embeddings have 3072).
- **Prompt:** the model is told to answer only from the given context and to say
  "I don't know based on the document" otherwise, which reduces hallucination.
- **Re-running ingest:** `ingest.py` empties the table first (`TRUNCATE`), so running it twice
  does not create duplicates. Use it only on a database dedicated to this project.
- **Free-tier limits:** ingest rejects files over 5 MB, over 100 pages, or producing more than
  300 chunks, before any API call is made.

## Test results

| Question                                | Result                                 |
| --------------------------------------- | -------------------------------------- |
| What was the total revenue?             | Correct (Rs 820 crore vs Rs 690 crore) |
| What dividend did the board recommend?  | Correct (Rs 4.50 per share)            |
| Who is the CEO of Google?               | "I don't know based on the document."  |
| What was the company's revenue in 2030? | "I don't know based on the document."  |

A question with a typo ("totla revenue") still retrieved the right chunks, because embeddings
match meaning rather than exact keywords.

## Known limitations

- Regex sentence splitting can leave stray section numbers (like "4.") at the end of a chunk.
- Retrieval always returns 4 chunks, even for unrelated questions. The "I don't know" comes
  from the prompt, not from a similarity threshold.
- Only text-based PDFs work. Scanned PDFs would need OCR.
- One document at a time: ingesting a new PDF replaces the previous one.
- Chunk size (500 characters) and the number of retrieved chunks were not tuned or evaluated.

## Possible improvements

- A distance threshold to skip the model call when nothing relevant is found
- Multiple documents, using a `document_id` column
- Reranking and a small set of test questions to measure answer quality
- A web interface (Streamlit, or React with a Python API)
