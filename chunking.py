def create_chunks(text, chunk_size=500):
    # Metni boş satırlardan böl.
    # Böylece her paragraf ayrı bir parça haline gelir.
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Baştaki ve sondaki gereksiz boşlukları temizle
        paragraph = paragraph.strip()

        # Boş paragraf varsa atla
        if not paragraph:
            continue

        # Paragraf mevcut chunk'a eklenince
        # chunk_size sınırını geçmiyorsa birlikte tut
        if len(current_chunk) + len(paragraph) <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:
            # Mevcut chunk dolduysa önce onu kaydet
            if current_chunk:
                chunks.append(current_chunk)

            # Yeni chunk'ı bu paragrafla başlat
            current_chunk = paragraph

    # Döngü bittikten sonra elimizde kalan son chunk'ı da kaydet
    if current_chunk:
        chunks.append(current_chunk)

    return chunks