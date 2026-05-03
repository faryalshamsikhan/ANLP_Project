import faiss
import pickle
import numpy as np

from rag.embedder import get_embedding
from evaluation.gpt_grader import gpt_grade
from evaluation.rag_grader import rag_grade
from evaluation.compare import compare


INDEX_FILE = "rag_index.faiss"
META_FILE = "rag_meta.pkl"


# Load once
index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)


def search(query, k=5):

    q_emb = get_embedding(query)
    q_emb = q_emb / np.linalg.norm(q_emb)

    _, idx = index.search(np.array([q_emb], dtype="float32"), k)

    return [metadata[i] for i in idx[0]]


if __name__ == "__main__":

    print("🚀 Exam Evaluator Ready (Offline RAG)\n")

    question = input("Enter question: ")
    answer = input("Enter student answer: ")

    retrieved = search(question)

    context = retrieved[0]  # best mWatch

    print("\n⏳ Evaluating...\n")

    gpt_result = gpt_grade(context, answer)
    rag_result = rag_grade(retrieved, question, answer)

    print(compare(gpt_result, rag_result))