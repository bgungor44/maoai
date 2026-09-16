import os
from pathlib import Path

import chromadb
import requests
from pypdf import PdfReader

from chunking import create_chunks


EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
KNOWLEDGE_DIR = Path("knowledge")
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "maoai_knowledge"
BATCH_SIZE = 32

HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Content-Type": "application/json",
}


def get_embeddings(texts):
    # NVIDIA embedding API'sine metinleri toplu gönder.
    data = {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": texts,
        "input_type": "passage",
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=HEADERS,
        json=data,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    return [item["embedding"] for item in result["data"]]


def read_source(path):
    # TXT ve Markdown dosyalarını doğrudan oku.
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    # PDF içindeki metni sayfa sayfa çıkar.
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)

        return "\n\n".join(pages)

    return ""


def load_documents():
    # knowledge klasöründeki desteklenen bütün kaynakları bul.
    supported = {".txt", ".md", ".pdf"}
    files = sorted(
        path
        for path in KNOWLEDGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )

    documents = []

    for path in files:
        text = read_source(path)
        chunks = create_chunks(text)

        for chunk_index, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:
                continue

            documents.append(
                {
                    "text": chunk,
                    "source": str(path),
                    "chunk_index": chunk_index,
                }
            )

    return files, documents


def main():
    files, documents = load_documents()

    if not files:
        raise SystemExit(
            "knowledge klasöründe .txt, .md veya .pdf kaynak bulunamadı."
        )

    if not documents:
        raise SystemExit("Kaynaklardan kullanılabilir metin çıkarılamadı.")

    print(f"{len(files)} kaynak dosya bulundu.")
    print(f"{len(documents)} chunk oluşturuldu.")

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Her ingest işleminde koleksiyonu yeniden kuruyoruz.
    # Böylece silinen/değiştirilen dosyalardan eski chunk kalmaz.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # Büyük kitaplarda API'ye binlerce chunk'ı tek istekte göndermemek için
    # embedding işlemini batch'ler halinde yap.
    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch = documents[batch_start:batch_start + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        embeddings = get_embeddings(texts)

        ids = [
            f"chunk_{batch_start + i}"
            for i in range(len(batch))
        ]
        metadatas = [
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"],
            }
            for item in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        done = min(batch_start + len(batch), len(documents))
        print(f"Embedding: {done}/{len(documents)}")

    print("Vector database hazır.")


if __name__ == "__main__":
    main()
