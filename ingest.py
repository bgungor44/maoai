import os
import re
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from chunking import create_chunks


# Proje kökündeki .env dosyasındaki API anahtarlarını ortam değişkenlerine yükle.
load_dotenv()

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


def read_text_source(path):
    # TXT ve Markdown dosyalarını doğrudan oku.
    return path.read_text(encoding="utf-8")


def decode_legacy_pdf_glyphs(text):
    # Bazı eski Türkçe PDF'lerde pypdf gerçek harfler yerine
    # /G61/G6F/G... biçiminde font glyph kodları döndürüyor.
    # Bu PDF'de kodların büyük kısmı Windows-1254 byte değerlerine karşılık geliyor:
    # /G61 -> a, /GFD -> ı, /GFE -> ş, /GF0 -> ğ, /GFC -> ü, /GF6 -> ö, /GE7 -> ç.
    # Chunk oluşturmadan ÖNCE bunları gerçek Türkçe karakterlere çeviriyoruz.
    if not re.search(r"/G[0-9A-Fa-f]{2}", text):
        return text

    def replace_glyph(match):
        value = int(match.group(1), 16)

        try:
            return bytes([value]).decode("cp1254")
        except UnicodeDecodeError:
            # Tanımsız bir byte gelirse veri kaybetmek yerine orijinal kodu bırak.
            return match.group(0)

    decoded = re.sub(r"/G([0-9A-Fa-f]{2})", replace_glyph, text)

    # Bazı fontlarda noktalama işaretleri özel glyph kodlarıyla geliyor.
    # Geriye kalan yaygın kodları okunabilir Unicode işaretlerine dönüştür.
    decoded = decoded.replace("/G92", "’")
    decoded = decoded.replace("/G93", "“")
    decoded = decoded.replace("/G94", "”")

    return decoded


def read_pdf_pages(path):
    # PDF'yi tek metin yapmak yerine sayfa sayfa oku.
    # Böylece her chunk'ın hangi PDF sayfasından geldiğini kaybetmeyiz.
    reader = PdfReader(str(path))
    pages = []
    repaired_pages = 0

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Font encoding'i bozuk eski PDF'leri mümkünse otomatik düzelt.
        if re.search(r"/G[0-9A-Fa-f]{2}", text):
            text = decode_legacy_pdf_glyphs(text)
            repaired_pages += 1

        text = text.strip()

        if not text:
            print(f"UYARI: {path} sayfa {page_number}: metin çıkarılamadı.")
            continue

        pages.append(
            {
                "text": text,
                "page": page_number,
            }
        )

    if repaired_pages:
        print(
            f"PDF font düzeltme: {path} içinde "
            f"{repaired_pages} sayfa glyph kodlarından çözüldü."
        )

    return pages


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
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # PDF'de her sayfayı ayrı chunk'la ki metadata'ya doğru sayfa yazılsın.
            pages = read_pdf_pages(path)

            for page_data in pages:
                chunks = create_chunks(page_data["text"])

                for chunk_index, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 20:
                        continue

                    documents.append(
                        {
                            "text": chunk,
                            "source": str(path),
                            "page": page_data["page"],
                            "chunk_index": chunk_index,
                        }
                    )

        else:
            # TXT/Markdown dosyalarında sayfa olmadığı için page=0 kullanıyoruz.
            text = read_text_source(path)
            chunks = create_chunks(text)

            for chunk_index, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue

                documents.append(
                    {
                        "text": chunk,
                        "source": str(path),
                        "page": 0,
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
                "page": item["page"],
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
