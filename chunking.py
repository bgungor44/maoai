def create_chunks(text, chunk_size=200, overlap=40):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks



with open("knowledge/bilgi.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = create_chunks(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)