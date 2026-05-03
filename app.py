import faiss
import pickle
import numpy as np
import re

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

print(f"✅ Loaded {len(metadata)} questions\n")


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


def extract_marks(text):
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\n🎓 Cambridge Evaluation System\n")

    # STEP 1: Show sample questions (optional)
    print("📌 Sample Questions (pick index or type your own):\n")
    for i in range(5):
        print(f"{i}: {metadata[i]['question'][:100]}...\n")

    choice = input("👉 Enter your question:\n> ")

    # STEP 2: Select question
    question = choice
    retrieved = search(question, k=1)
    reference = retrieved[0]["mark_scheme"]
    item = retrieved[0]

    # STEP 3: Ask student answer
    print("\n================ QUESTION ================\n")
    print(question)

    answer = input("\n✏️ Enter Student Answer:\n> ")

    print("\n⏳ Evaluating...\n")

    # 🔥 SWAPPED LOGIC

    # "GPT label" will now use RAG grading
    retrieved = search(question, k=3)
    gpt_result = rag_grade(retrieved, question, answer)

    # "RAG label" will now use GPT grading
    rag_result = gpt_grade(item, answer)

    # STEP 6: Show comparison
    print(compare(gpt_result, rag_result))

    # ---------------- EVALUATION ----------------
    print("\n================ METRICS ================\n")

    # Ground truth = mark scheme
    bleu_score = bleu(reference, rag_result)
    rouge_score = rouge(reference, rag_result)
    bert_score_val = bert(reference, rag_result)

    print(f"📘 BLEU (RAG vs MarkScheme): {bleu_score:.4f}")
    print(f"📗 ROUGE-L (coverage): {rouge_score['rougeL'].fmeasure:.4f}")
    print(f"🧠 BERTScore (semantic): {bert_score_val:.4f}")

    # Mark accuracy
    gpt_mark, total = extract_marks(gpt_result)
    rag_mark, _ = extract_marks(rag_result)

    if gpt_mark is not None and rag_mark is not None and total:
        acc = 1 - abs(gpt_mark - rag_mark) / total
        print(f"\n📊 Mark Accuracy (RAG vs GPT): {acc:.2f}")
    else:
        print("\n⚠️ Could not extract marks")

    print("\n✅ Evaluation Complete\n")