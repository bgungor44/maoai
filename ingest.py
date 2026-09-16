import os
import requests
import chromadb

from chunking import create_chunks


EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Content-Type": "application/json",
}


def get_embeddings(texts):
    data = {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": texts,
        "input_type": "passage",
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=HEADERS,
        json=data,
    )

    response.raise_for_status()

    result = response.json()

    return [item["embedding"] for item in result["data"]]


with open("knowledge/bilgi.txt", "r", encoding="utf-8") as file:
    text = file.read()


chunks = create_chunks(text)

# Aşırı küçük chunk'ları şimdilik alma
chunks = [
    chunk
    for chunk in chunks
    if len(chunk.strip()) >= 20
]


print(f"{len(chunks)} chunk oluşturuldu.")


embeddings = get_embeddings(chunks)


client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = client.get_or_create_collection(
    name="maoai_knowledge"
)


ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
)


print(f"{len(chunks)} chunk vector database'e kaydedildi.")