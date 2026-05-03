from ingestion.pair_builder import build_pairs
from rag.store import RAGStore
from evaluation.gpt_grader import gpt_grade
from evaluation.rag_grader import rag_grade
from evaluation.compare import compare

import random

# Folders
QP_FOLDER = "cambridge_cs_evaluator/data/question_papers"
MS_FOLDER = "cambridge_cs_evaluator/data/mark_schemes"


if __name__ == "__main__":

    print("🚀 Starting Cambridge Evaluator...\n")

    # STEP 1: Load dataset
    dataset = build_pairs(QP_FOLDER, MS_FOLDER)
    print(f"✅ Loaded pairs: {len(dataset)}")

    # STEP 2: Build RAG index
    store = RAGStore()
    store.build(dataset)
    print("✅ RAG index built\n")

    # STEP 3: Select question
    context = random.choice(dataset)
    question = context["question"]

    print("\n================ QUESTION ================\n")
    print(question)

    # STEP 4: Student answer
    answer = input("\n✏️ Enter your answer here: ").strip()

    print("\n⏳ Evaluating...\n")

    # STEP 5: GPT grading (baseline)
    gpt_result = gpt_grade(context, answer)

    # STEP 6: RAG retrieval (IMPORTANT FIX)
    retrieved_chunks = store.search(question, k=5)

    # STEP 7: RAG grading (FIXED)
    rag_result = rag_grade(
        retrieved_chunks,
        question,
        answer
    )

    # STEP 8: Compare results
    print(compare(gpt_result, rag_result))