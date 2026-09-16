def _split_long_text(text, chunk_size, overlap):
    # Tek bir paragraf chunk_size'dan uzunsa onu da parçalara ayır.
    # overlap sayesinde iki parça arasında biraz ortak metin kalır.
    pieces = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return pieces


def create_chunks(text, chunk_size=1000, overlap=150):
    # Metni boş satırlardan böl.
    # Böylece mümkün olduğunca paragraf bütünlüğünü koruruz.
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Baştaki ve sondaki gereksiz boşlukları temizle
        paragraph = paragraph.strip()

        # Boş paragraf varsa atla
        if not paragraph:
            continue

        # Çok uzun tek bir paragraf varsa önce mevcut chunk'ı kaydet
        # sonra uzun paragrafı kendi içinde böl.
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.extend(
                _split_long_text(paragraph, chunk_size, overlap)
            )
            continue

        separator_size = 2 if current_chunk else 0

        # Paragraf mevcut chunk'a eklenince sınırı geçmiyorsa birlikte tut
        if len(current_chunk) + separator_size + len(paragraph) <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:
            # Mevcut chunk dolduysa önce onu kaydet
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    # Döngü bittikten sonra elimizde kalan son chunk'ı da kaydet
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
