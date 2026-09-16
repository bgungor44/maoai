# MAOAI

MAOAI, yalnızca kullanıcının `knowledge/` klasörüne eklediği kaynaklardan cevap üretmek için geliştirilen kaynak-temelli bir RAG asistanıdır.

## Nasıl çalışıyor?

1. `ingest.py`, `knowledge/` içindeki TXT, Markdown ve PDF dosyalarını okur.
2. Metinler `chunking.py` ile küçük parçalara ayrılır.
3. NVIDIA Nemotron embedding modeli her parçayı vektöre çevirir.
4. Vektörler ve kaynak bilgileri yerel ChromaDB veritabanına kaydedilir.
5. Kullanıcı soru sorduğunda soru da embedding'e çevrilir.
6. ChromaDB soruya en yakın kaynak parçalarını bulur.
7. Groq üzerindeki `openai/gpt-oss-20b`, yalnızca bulunan parçaları kullanarak Türkçe cevap üretir.
8. Kaynaklarda cevap yoksa modelden `Bilmiyorum.` demesi istenir.

## Kurulum

```bash
pip install -r requirements.txt
```

API anahtarlarını terminal ortamına ekle:

```bash
export NVIDIA_API_KEY="..."
export GROQ_API_KEY="..."
```

API anahtarlarını repoya veya kaynak koduna yazma.

## Kaynak ekleme

Kaynaklarını `knowledge/` klasörüne koyabilirsin.

Desteklenen formatlar:

- `.txt`
- `.md`
- `.pdf`

Örnek:

```text
knowledge/
├── bilgi.txt
├── mao_biyografi.pdf
├── gerilla_savasi.pdf
└── notlar.md
```

Yeni bir kaynak eklediğinde veya mevcut kaynağı değiştirdiğinde veritabanını yeniden oluştur:

```bash
python3 ingest.py
```

`vector_db/` otomatik oluşur ve Git tarafından takip edilmez.

## MAOAI'yi çalıştırma

```bash
python3 main.py
```

veya doğrudan:

```bash
python3 rag.py
```

Program açık kaldığı sürece art arda soru sorabilirsin. Çıkmak için `çık` yaz.

Her cevapta retrieval sırasında kullanılan kaynak dosyası ve distance değeri terminalde gösterilir. Cevap içinde kullanılan bilgiler `[KAYNAK 1]` biçiminde işaretlenir.

## Testler

```bash
python3 -m unittest discover -s tests -v
```

Chunking testleri GitHub Actions üzerinden her push ve pull request'te otomatik çalışacak şekilde ayarlanmıştır.

## Önemli sınır

MAOAI'nin "yalnızca kaynaklardan konuşması" iki katmanla sağlanır: önce ChromaDB ilgili kaynakları seçer, ardından LLM'e yalnızca bu kaynakları kullanması söylenir. Bir LLM'in talimatlara yüzde 100 uyacağı matematiksel olarak garanti edilemez. Bu nedenle ilerleyen aşamada retrieval değerlendirme testleri ve gerektiğinde ek doğrulama katmanları kullanılmalıdır.
