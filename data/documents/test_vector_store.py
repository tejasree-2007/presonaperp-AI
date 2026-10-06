
from loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import store_embeddings


pdf_path = "data/documents/resume.pdf"

text = load_document(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

store_embeddings(chunks, embeddings)

print("Successfully stored", len(chunks), "chunks in ChromaDB.")
