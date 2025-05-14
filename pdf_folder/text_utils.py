# text_utils.py
import re

def clean_text(text):
    """Clean text by removing unwanted characters and extra spaces."""
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\$.*?\$", "", text)  # Remove LaTeX math
    text = re.sub(r"\[[^\]]*\]", "", text)  # Remove references in square brackets
    return text.strip()

def chunk_text(text, chunk_size=300, overlap=50):
    """Chunk the text into smaller parts with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
