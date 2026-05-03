def extract_text(pdf_path):
    import pdfplumber
    import re

    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()

            if t:
                # remove encoding noise only
                t = re.sub(r"\(cid:.*?\)", "", t)

                # DO NOT flatten structure
                t = re.sub(r"[ \t]+", " ", t)

                pages.append(t.strip())

    text = "\n\n".join(pages)

    # remove header/footer noise (important for Cambridge papers)
    text = re.sub(r"Cambridge.*?Paper", "", text, flags=re.IGNORECASE)
    text = re.sub(r"INSTRUCTIONS.*?INFORMATION", "", text, flags=re.IGNORECASE)

    return text.strip()