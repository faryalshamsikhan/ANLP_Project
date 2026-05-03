from config import client, LLM_MODEL

def rag_grade(retrieved_chunks, question, answer):

    # 🔥 Combine retrieved mark schemes (not chunks anymore)
    context = "\n\n".join(
        [item["mark_scheme"] for item in retrieved_chunks]
    )

    prompt = f"""
STRICT RUBRIC GRADING

Question:
{question}

Relevant Marking Scheme:
{context}

Student Answer:
{answer}

Rules:
- Follow marking scheme strictly
- Award marks point by point
- Mention missing points clearly
- Give final marks like: X/Y
"""

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content