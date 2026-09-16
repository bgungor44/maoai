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


# ChromaDB'de en yakın chunk'ı ara
results = collection.query(
    query_embeddings=[soru_vector],
    n_results=1,
)


best_chunk = results["documents"][0][0]
distance = results["distances"][0][0]


print("\n----------------------")
print("BULUNAN KAYNAK:")
print(best_chunk)
print("Distance:", distance)
print("----------------------")


# Bulunan kaynağı LLM'e gönder
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
