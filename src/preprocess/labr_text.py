import os
import re
import pyarabic.araby as araby

def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""
    # Suppression tatweel (kashida)
    text = araby.strip_tatweel(text)
    # Suppression diacritiques (harakat)
    text = araby.strip_tashkeel(text)
    # Normalisation alif
    text = re.sub("[إأآا]", "ا", text)
    # Normalisation yaa / alif maqsoura
    text = re.sub("ى", "ي", text)
    # Normalisation taa marbuta
    text = re.sub("ة", "ه", text)
    # Remplacement retours à la ligne et espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

