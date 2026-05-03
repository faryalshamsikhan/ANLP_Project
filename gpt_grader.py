from config import client, LLM_MODEL

def gpt_grade(context, answer):

    prompt = f"""
You are an ICT examiner.

Question:
{context['question']}

Mark Scheme:
{context['mark_scheme']}

Student Answer:
{answer}

Give marks and justification.
"""

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content