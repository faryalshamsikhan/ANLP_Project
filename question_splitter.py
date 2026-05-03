import re

def split_questions(text):

    # Step 1: normalize spaces but keep structure
    text = re.sub(r"\s+", " ", text)

    # Step 2: force separation before question numbers
    text = re.sub(r"(\d{1,2}\s*\([a-z]\))", r"\n\1", text)  # (a), (b)
    text = re.sub(r"(?<=\D)(\d{1,2}\s+[a-zA-Z])", r"\n\1", text)

    # Step 3: split on question start patterns
    parts = re.split(r"\n(?=\d{1,2}\s)", text)

    cleaned = []

    for p in parts:
        p = p.strip()

        # filter out headers/instructions
        if len(p) < 80:
            continue

        # must contain bracket marks OR question pattern
        if "[" in p or "(" in p:
            cleaned.append(p)

    return cleaned