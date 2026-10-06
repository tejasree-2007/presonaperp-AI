from loader import load_document
from rag.chunker import chunk_text

pdf_path = "data/documents/resume.pdf"

text = load_document(pdf_path)

chunks = chunk_text(text)

print("Total chunks:", len(chunks))

print("\n--- FIRST CHUNK ---\n")
print(chunks[0])