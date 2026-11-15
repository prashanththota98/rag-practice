import os
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
MAX_QUESTIONS = 10        # per visitor session
MAX_QUESTION_CHARS = 300

client = genai.Client()

st.title("Document Q&A (RAG)")
st.caption("Ask questions about a fictional annual report. Built with Gemini and Postgres pgvector.")

if "count" not in st.session_state:
    st.session_state.count = 0

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    if st.session_state.count >= MAX_QUESTIONS:
        st.warning("Question limit reached for this session.")
    elif len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep the question under {MAX_QUESTION_CHARS} characters.")
    else:
        st.session_state.count += 1
        try:
            # 1. embed the question
            result = client.models.embed_content(model=EMBED_MODEL, contents=question)
            q_vec = result.embeddings[0].values

            # 2. retrieve the 4 closest chunks
            conn = psycopg2.connect(os.environ["DATABASE_URL"])
            cur = conn.cursor()
            cur.execute(
                "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT 4",
                (str(q_vec),)
            )
            retrieved = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()

            # 3. ask the model, using only the retrieved text
            context = "\n\n".join(retrieved)
            prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
Answer:"""
            response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)

            st.subheader("Answer")
            st.write(response.text)

            with st.expander("Sources"):
                for i, chunk in enumerate(retrieved, 1):
                    st.write(f"{i}. {chunk}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            