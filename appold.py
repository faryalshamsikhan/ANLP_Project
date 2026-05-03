import random
import faiss
import pickle
import numpy as np

from rag.embedder import get_embedding
from evaluation.gpt_grader import gpt_grade
from evaluation.rag_grader import rag_grade
from evaluation.compare import compare

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score


INDEX_FILE = "rag_index.faiss"
META_FILE = "rag_meta.pkl"


# ---------------- LOAD ----------------
print("🚀 Loading system...")

index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

print("✅ Ready!\n")


# ---------------- SEARCH ----------------
def search(query, k=3):

    emb = get_embedding([query])[0]
    emb = np.array(emb, dtype="float32")
    emb = emb / np.linalg.norm(emb)

    _, idx = index.search(np.array([emb]), k)

    return [metadata[i] for i in idx[0]]


# ---------------- METRICS ----------------
def bleu(ref, hyp):
    return sentence_bleu([ref.split()], hyp.split())


def rouge(ref, hyp):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    return scorer.score(ref, hyp)


def bert(ref, hyp):
    P, R, F1 = bert_score.score([hyp], [ref], lang="en", verbose=False)
    return float(F1[0])


# ---------------- MAIN ----------------
if __name__ == "__main__":

    item = random.choice(metadata)
    question = item["question"]

    print("\n================ QUESTION ================\n")
    print(question)

    answer = input("\n✏️ Student Answer: ")

    print("\n⏳ Evaluating...\n")

    # GPT grading
    gpt_result = gpt_grade(item, answer)

    # RAG retrieval
    retrieved = search(question, k=3)

    # RAG grading
    rag_result = rag_grade(retrieved, question, answer)

    print(compare(gpt_result, rag_result))

    print("\n================ METRICS ================\n")

    print("BLEU:", bleu(gpt_result, rag_result))
    print("ROUGE:", rouge(gpt_result, rag_result)["rougeL"].fmeasure)
    print("BERT:", bert(gpt_result, rag_result))

    print("\n✅ Done")