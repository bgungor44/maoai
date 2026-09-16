import os
from openai import OpenAI


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)


with open("knowledge/bilgi.txt", "r", encoding="utf-8") as file:
    bilgi = file.read()


soru = input("Sen: ")


response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[
        {
            "role": "system",
            "content": f"""
Sen MAOAI isimli bir bilgi asistanısın.

Aşağıda sana KAYNAK verilecek.

Kurallar:
- Soruları SADECE verilen KAYNAK içerisindeki bilgilere dayanarak cevapla.
- Kendi önceden sahip olduğun bilgileri kullanma.
- Kaynakta soruyu cevaplamak için yeterli bilgi yoksa sadece:
  "Bilmiyorum."
  de.
- Cevabı Türkçe ver.

KAYNAK:
{bilgi}
"""
        },
        {
            "role": "user",
            "content": soru
        }
    ],
    temperature=0,
)

print("\nMAOAI:", response.choices[0].message.content)