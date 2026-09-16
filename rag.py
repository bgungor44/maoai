import os

import chromadb
import requests
from dotenv import load_dotenv
from openai import OpenAI


# Proje kökündeki .env dosyasındaki API anahtarlarını ortam değişkenlerine yükle.
load_dotenv()

EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "maoai_knowledge"
TOP_K = 5

HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Content-Type": "application/json",
}


def get_query_embedding(text):
    # Kullanıcının sorusunu arama için vektöre çevir.
    data = {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": [text],
        "input_type": "query",
    }

    response = requests.post(
        EMBEDDING_URL,
        headers=HEADERS,
        json=data,
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


def get_collection():
    # Diskteki ChromaDB'ye bağlan.
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        raise SystemExit(
            "Vector database bulunamadı. Önce 'python3 ingest.py' çalıştır."
        ) from error


def retrieve(collection, question):
    # Soruyu embedding'e çevir ve en yakın chunk'ları bul.
    question_vector = get_query_embedding(question)
    result_count = min(TOP_K, collection.count())

    if result_count == 0:
        return []

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=result_count,
        include=["documents", "distances", "metadatas"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return list(zip(documents, distances, metadatas))


def build_context(matches):
    # LLM her chunk'ın hangi dosyadan geldiğini de görsün.
    parts = []

    for index, (chunk, distance, metadata) in enumerate(matches, start=1):
        source = metadata.get("source", "bilinmeyen kaynak")
        parts.append(
            f"[KAYNAK {index} | dosya: {source}]\n{chunk}"
        )

    return "\n\n".join(parts)


def print_matches(matches):
    # Geliştirme sırasında retrieval sonucunu görebilmek için terminale yaz.
    for index, (chunk, distance, metadata) in enumerate(matches, start=1):
        print(f"\n--- KAYNAK {index} ---")
        print(f"Dosya: {metadata.get('source', 'bilinmeyen kaynak')}")
        print(f"Distance: {distance:.4f}")
        print(chunk)


def answer_question(client, collection, question):
    matches = retrieve(collection, question)

    if not matches:
        return "Bilmiyorum."

    print_matches(matches)
    context = build_context(matches)

    # Groq üzerindeki model sadece retrieval ile gelen kaynakları görerek cevap üretir.
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
Sen MAOAI isimli kaynak-temelli bir bilgi asistanısın.

Zorunlu kurallar:
- Yalnızca kullanıcı mesajında verilen KAYNAKLAR'daki bilgileri kullan.
- Kendi eğitim verindeki bilgileri, genel kültürünü veya tahminlerini kullanma.
- Kaynaklarda sorunun cevabını destekleyen açık bilgi yoksa yalnızca "Bilmiyorum." yaz.
- Kaynaklarda kısmi bilgi varsa yalnızca desteklenen kısmı söyle; eksik kısmı uydurma.
- Kaynaklar birbiriyle çelişiyorsa bunu açıkça belirt ve iki bilgiyi de kaynak numarasıyla göster.
- Cevap verirken kullandığın bilgilerin sonuna [KAYNAK 1] gibi kaynak numarası ekle.
- Cevabı Türkçe ver.
""",
            },
            {
                "role": "user",
                "content": f"""
KAYNAKLAR:
{context}

SORU:
{question}
""",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def main():
    collection = get_collection()

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )

    print("MAOAI hazır. Çıkmak için 'çık' yaz.\n")

    # Programı her soru için yeniden başlatmak yerine sohbet döngüsünde tut.
    while True:
        try:
            question = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz.")
            break

        if not question:
            continue

        if question.lower() in {"çık", "cik", "exit", "quit"}:
            print("Görüşürüz.")
            break

        try:
            answer = answer_question(client, collection, question)
            print(f"\nMAOAI: {answer}\n")
        except requests.RequestException as error:
            print(f"\nEmbedding API hatası: {error}\n")
        except Exception as error:
            print(f"\nMAOAI hata verdi: {error}\n")


if __name__ == "__main__":
    main()
