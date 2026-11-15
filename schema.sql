CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(3072)
);