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
DOCUMENT_SAMPLE_SIZE = 24
DOCUMENT_BATCH_SIZE = 6

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


def source_label(metadata):
    # PDF ise dosya adına ek olarak sayfa numarasını da göster.
    source = metadata.get("source", "bilinmeyen kaynak")
    page = metadata.get("page", 0)

    if page:
        return f"{source}, sayfa {page}"

    return source


def build_context(matches):
    # LLM her chunk'ın hangi dosyadan ve PDF ise hangi sayfadan geldiğini görsün.
    parts = []

    for index, (chunk, distance, metadata) in enumerate(matches, start=1):
        label = source_label(metadata)
        parts.append(f"[KAYNAK {index} | {label}]\n{chunk}")

    return "\n\n".join(parts)


def print_matches(matches):
    # Geliştirme sırasında retrieval sonucunu görebilmek için terminale yaz.
    for index, (chunk, distance, metadata) in enumerate(matches, start=1):
        print(f"\n--- KAYNAK {index} ---")
        print(f"Dosya: {metadata.get('source', 'bilinmeyen kaynak')}")

        page = metadata.get("page", 0)
        if page:
            print(f"Sayfa: {page}")

        print(f"Distance: {distance:.4f}")
        print(chunk)


def is_document_wide_question(question):
    # Belgenin tamamı/geneli hakkında cevap gerektiren ifadeleri yakala.
    # Noktasal sorular eski Top-K RAG yolunda kalmaya devam eder.
    text = question.casefold()
    markers = (
        "pdf'yi özetle", "pdfyi özetle", "pdf'i özetle", "pdfi özetle",
        "pdf genel olarak", "pdf'de genel olarak", "pdfde genel olarak",
        "belgeyi özetle", "belge genel olarak", "belgede genel olarak",
        "kitabı özetle", "kitap genel olarak", "kitapta genel olarak",
        "dokümanı özetle", "dokumanı özetle", "doküman genel olarak",
        "bu pdf", "bu belge", "bu kitap", "bu doküman", "bu dokuman",
        "genel bir özet", "genel özet",
    )
    return any(marker in text for marker in markers)


def get_document_samples(collection):
    # ChromaDB'deki tüm chunk metadata ve metinlerini al.
    # Sonra her kaynak dosyayı baştan sona temsil edecek şekilde eşit aralıklı
    # örnekler seç. Böylece sadece embedding'in en yakın 5 sonucuna bakmayız.
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    by_source = {}
    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "bilinmeyen kaynak")
        by_source.setdefault(source, []).append((document, metadata))

    samples = []

    for source, items in by_source.items():
        # PDF'lerde sayfa sırasına, diğer kaynaklarda chunk sırasına göre diz.
        items.sort(
            key=lambda item: (
                item[1].get("page", 0),
                item[1].get("chunk_index", 0),
            )
        )

        sample_count = min(DOCUMENT_SAMPLE_SIZE, len(items))
        if sample_count == 1:
            selected_indexes = [0]
        else:
            selected_indexes = [
                round(i * (len(items) - 1) / (sample_count - 1))
                for i in range(sample_count)
            ]

        for index in selected_indexes:
            document, metadata = items[index]
            samples.append((document, metadata))

    return samples


def summarize_sample_batch(client, question, batch):
    # Belgenin küçük bir bölümünü analiz edip yalnızca soruyla ilgili notları çıkar.
    parts = []
    for chunk, metadata in batch:
        parts.append(f"[{source_label(metadata)}]\n{chunk}")

    context = "\n\n".join(parts)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
Sen belge analizi yapan bir ara aşamasın.
Yalnızca verilen METİN PARÇALARI'nı kullan.
Kullanıcının isteği açısından önemli olan ana fikirleri kısa notlar halinde çıkar.
Metinde olmayan bilgi ekleme.
Her notta mümkünse dosya ve sayfa bilgisini koru.
İlgili bilgi yoksa "İlgili bilgi yok." yaz.
Cevabı Türkçe ver.
""",
            },
            {
                "role": "user",
                "content": f"""
KULLANICI İSTEĞİ:
{question}

METİN PARÇALARI:
{context}
""",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def answer_document_wide_question(client, collection, question):
    # Map-reduce benzeri belge analizi:
    # 1) Belgenin farklı noktalarından örnek chunk'lar al.
    # 2) Küçük grupları ayrı ayrı analiz et.
    # 3) Ara notları tek bir nihai cevapta birleştir.
    samples = get_document_samples(collection)

    if not samples:
        return "Bilmiyorum."

    print(f"\n[BELGE ANALİZİ] {len(samples)} temsilci chunk inceleniyor...")

    notes = []
    for start in range(0, len(samples), DOCUMENT_BATCH_SIZE):
        batch = samples[start:start + DOCUMENT_BATCH_SIZE]
        note = summarize_sample_batch(client, question, batch)
        notes.append(note)
        print(f"Belge analizi: {min(start + len(batch), len(samples))}/{len(samples)}")

    combined_notes = "\n\n--- ARA NOT ---\n\n".join(notes)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": """
Sen MAOAI isimli kaynak-temelli bir bilgi asistanısın.
Aşağıdaki ARA NOTLAR, belgenin farklı bölümlerinden çıkarılmıştır.
Yalnızca bu notlarda desteklenen bilgileri kullan.
Belgenin tamamını eksiksiz okuduğunu veya her sayfayı doğruladığını iddia etme.
Notlarda yeterli bilgi yoksa "Bilmiyorum." de.
Çelişkili notlar varsa çelişkiyi açıkça belirt.
Dosya/sayfa bilgisi bulunan önemli iddialarda bu bilgiyi koru.
Cevabı Türkçe ver.
""",
            },
            {
                "role": "user",
                "content": f"""
KULLANICI İSTEĞİ:
{question}

ARA NOTLAR:
{combined_notes}
""",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def answer_question(client, collection, question):
    # Belge-geneli soruları normal Top-5 retrieval yerine ayrı analiz yoluna gönder.
    if is_document_wide_question(question):
        return answer_document_wide_question(client, collection, question)

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
