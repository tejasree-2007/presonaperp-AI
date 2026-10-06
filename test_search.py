from loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import search


pdf_path = "data/documents/resume.pdf"

text = load_document(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

query = "What are my technical skills?"

query_embedding = create_embeddings([query])[0]

results = search(query_embedding)

print("\n--- SEARCH RESULTS ---\n")

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(result)
    print("\n" + "-" * 50)