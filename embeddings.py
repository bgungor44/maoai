import os
import requests
import math


URL = "https://integrate.api.nvidia.com/v1/embeddings"

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

    response = requests.post(URL, headers=HEADERS, json=data)
    response.raise_for_status()

    return response.json()["data"][0]["embedding"]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (length_a * length_b)


metinler = [
    "Ahmet'in kedisinin adı Zeytin'dir.",
    "Ahmet'in en sevdiği yemek mantıdır.",
    "Ahmet 2020 yılında yazılım öğrenmeye başlamıştır.",
]

soru = "Ahmet'in evcil hayvanının adı nedir?"


soru_vector = get_embedding(soru, "query")

for metin in metinler:
    metin_vector = get_embedding(metin, "passage")

    similarity = cosine_similarity(soru_vector, metin_vector)

    print(f"{similarity:.4f} -> {metin}")