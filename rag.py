import os
import requests
import chromadb

from openai import OpenAI


EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Content-Type": "application/json",
}


def get_query_embedding(text):
    data = {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": [text],
        "input_type": "query",
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=HEADERS,
        json=data,
    )

    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


# ChromaDB'ye bağlan
chroma_client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = chroma_client.get_collection(
    name="maoai_knowledge"
)


# Kullanıcıdan soru al
soru = input("Sen: ")


# Sadece soruyu embedding'e çevir
soru_vector = get_query_embedding(soru)


# ChromaDB'de soruya en yakın 3 chunk'ı ara
results = collection.query(
    query_embeddings=[soru_vector],
    n_results=3,
)


# ChromaDB'nin bulduğu chunk'ları ve distance değerlerini al
chunks = results["documents"][0]
distances = results["distances"][0]


# Bulunan 3 kaynağı terminalde göster
for i, (chunk, distance) in enumerate(
    zip(chunks, distances),
    start=1
):
    print(f"\n--- KAYNAK {i} ---")
    print(f"Distance: {distance}")
    print(chunk)


# Bulunan 3 chunk'ı tek bir metin haline getir
# Böylece LLM sadece tek chunk yerine 3 kaynağı birden görecek
context = "\n\n".join(chunks)


# Bulunan kaynakları Cerebras üzerindeki LLM'e gönder
# NVIDIA şimdilik sadece embedding üretmek için kullanılıyor
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ["CEREBRAS_API_KEY"],
)


response = client.chat.completions.create(
    model="qwen-3.8-27b",
    messages=[
        {
            "role": "system",
            "content": """
Sen MAOAI isimli bir bilgi asistanısın.

Sadece sana verilen KAYNAKLAR içerisindeki
bilgileri kullan.

Kaynaklar soruyu cevaplamak için yeterli değilse
"Bilmiyorum." de.

Kendi bilgilerini kullanma.
Cevabı Türkçe ver.
"""
        },
        {
            "role": "user",
            "content": f"""
KAYNAKLAR:
{context}

SORU:
{soru}
"""
        }
    ],
    temperature=0,
)


# LLM'in oluşturduğu cevabı kullanıcıya göster
print("\nMAOAI:", response.choices[0].message.content)
