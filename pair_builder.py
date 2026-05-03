import os
from ingestion.extract import extract_text
from ingestion.question_splitter import split_questions

def get_key(filename):
    """
    Robust key extraction for Cambridge papers
    Example: 2210_s20_qp_11.pdf → s20_11
    """
    name = filename.replace(".pdf", "")
    parts = name.split("_")

    if len(parts) < 3:
        return name  # fallback safety

    return f"{parts[1]}_{parts[-1]}"


def build_pairs(qp_folder, ms_folder):

    qp_files = {get_key(f): f for f in os.listdir(qp_folder)}
    ms_files = {get_key(f): f for f in os.listdir(ms_folder)}

    dataset = []

    matched = 0
    missing = 0

    for key in qp_files:

        if key not in ms_files:
            print(f"⚠️ Missing mark scheme for: {key}")
            missing += 1
            continue

        qp_path = os.path.join(qp_folder, qp_files[key])
        ms_path = os.path.join(ms_folder, ms_files[key])

        qp_text = extract_text(qp_path)
        ms_text = extract_text(ms_path)

        questions = split_questions(qp_text)

        # IMPORTANT IMPROVEMENT:
        # Instead of one big mark scheme → attach per question chunk
        for q in questions:

            dataset.append({
                "id": key,
                "question": q,
                "mark_scheme": ms_text  # later you can improve to per-question mapping
            })

        matched += 1

    print(f"\n✅ Matched papers: {matched}")
    print(f"⚠️ Missing pairs: {missing}")
    print(f"📦 Total Q-A entries: {len(dataset)}")

    return dataset