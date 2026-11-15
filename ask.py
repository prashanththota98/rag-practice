import os
import psycopg2
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

question = input("Ask a question: ")


result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)
q_vec = result.embeddings[0].values


conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute(
    "SELECT content FROM chunks ORDER BY embedding <=> %s::vector LIMIT 4",
    (str(q_vec),)
)
retrieved = [row[0] for row in cur.fetchall()]
cur.close()
conn.close()

context = "\n\n".join(retrieved)


prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
Answer:"""


response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print("\nANSWER:")
print(response.text)

print("\nSOURCES:")
for i, chunk in enumerate(retrieved, 1):
    print(f"{i}. {chunk[:100]}...")