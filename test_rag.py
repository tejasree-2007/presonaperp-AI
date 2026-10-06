from loader import load_document

pdf_path = "data/documents/resume.pdf"

text = load_document(pdf_path)

print("\n--- PDF TEXT ---\n")
print(text[:2000])