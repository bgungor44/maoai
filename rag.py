import os
import math
import requests

from openai import OpenAI
from chunking import create_chunks


EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Content-Type": "application/json",
}


def get_embedding(text, input_type):
    data = {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": [text],
        "input_type": input_type,
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=HEADERS,
        json=data
    )

    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (length_a * length_b)


with open("knowledge/bilgi.txt", "r", encoding="utf-8") as file:
    text = file.read()


chunks = create_chunks(text)

soru = input("Sen: ")

soru_vector = get_embedding(soru, "query")


best_chunk = None
best_score = -1


for chunk in chunks:

    chunk_vector = get_embedding(chunk, "passage")

    score = cosine_similarity(
        soru_vector,
        chunk_vector
    )

    print(f"\nScore: {score:.4f}")
    print(f"Chunk: {chunk}")

    if score > best_score:
        best_score = score
        best_chunk = chunk


print("\n----------------------")
print("EN ALAKALI CHUNK:")
print(best_chunk)
print("Similarity:", best_score)
print("----------------------")


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)


response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[
        {
            "role": "system",
            "content": """
Sen MAOAI isimli bir bilgi asistanısın.

Sadece sana verilen KAYNAK içerisindeki
bilgileri kullan.

Kaynak soruyu cevaplamak için yeterli değilse
"Bilmiyorum." de.

Kendi bilgilerini kullanma.
Cevabı Türkçe ver.
"""
        },
        {
            "role": "user",
            "content": f"""
KAYNAK:
{best_chunk}

SORU:
{soru}
"""
        }
    ],
    temperature=0,
)


print("\nMAOAI:", response.choices[0].message.content)