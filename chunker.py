import re

def chunk_text(text, max_words=250, overlap=50):

    # Step 1: normalize spaces
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + max_words
        chunk = " ".join(words[start:end])

        chunks.append(chunk.strip())

        # overlap for context continuity
        start = end - overlap

    return chunks