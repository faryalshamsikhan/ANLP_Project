import os
import faiss
import pickle
import numpy as np

from ingestion.extract import extract_text
from ingestion.question_splitter import split_questions
from rag.embedder import get_embedding

QP_FOLDER = "cambridge_cs_evaluator/data/question_papers"
MS_FOLDER = "cambridge_cs_evaluator/data/mark_schemes"

INDEX_FILE = "rag_index.faiss"
META_FILE = "rag_meta.pkl"


def get_key(filename):
    parts = filename.replace(".pdf", "").split("_")
    return f"{parts[1]}_{parts[-1]}"


def build_dataset():

    qp_files = {get_key(f): f for f in os.listdir(QP_FOLDER)}
    ms_files = {get_key(f): f for f in os.listdir(MS_FOLDER)}

    dataset = []

    for key in qp_files:

        if key not in ms_files:
            continue

        qp_text = extract_text(os.path.join(QP_FOLDER, qp_files[key]))
        ms_text = extract_text(os.path.join(MS_FOLDER, ms_files[key]))

        questions = split_questions(qp_text)

        for q in questions:
            dataset.append({
                "question": q,
                "mark_scheme": ms_text
            })

    return dataset


def build_index():

    dataset = build_dataset()
    print(f"📦 Total questions: {len(dataset)}")

    # 🔥 Embed ONLY questions (NO chunking)
    questions = [item["question"] for item in dataset]

    embeddings = get_embedding(questions)  # 🔥 BATCH CALL

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)

    metadata = []

    for emb, item in zip(embeddings, dataset):

        emb = np.array(emb, dtype="float32")
        emb = emb / np.linalg.norm(emb)

        index.add(np.array([emb]))

        metadata.append({
            "question": item["question"],
            "mark_scheme": item["mark_scheme"]
        })

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ FAST index built and saved!")


if __name__ == "__main__":
    build_index()