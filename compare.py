import re

def extract_marks(text):
    """
    Extracts marks like:
    Marks: 7/10
    """
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def compare(gpt, rag):

    gpt_score, gpt_total = extract_marks(gpt)
    rag_score, rag_total = extract_marks(rag)

    diff = None
    if gpt_score is not None and rag_score is not None:
        diff = abs(gpt_score - rag_score)

    analysis = f"""
================ GPT GRADING ================
{gpt}

================ RAG GRADING ================
{rag}

================ ANALYSIS ===================

"""

    if diff is not None:
        analysis += f"""
- GPT Score: {gpt_score}/{gpt_total}
- RAG Score: {rag_score}/{rag_total}
- Absolute Difference: {diff} marks
"""

        if diff == 0:
            analysis += "- Agreement: PERFECT MATCH\n"
        elif diff <= 2:
            analysis += "- Agreement: HIGH\n"
        elif diff <= 5:
            analysis += "- Agreement: MODERATE\n"
        else:
            analysis += "- Agreement: LOW (large discrepancy)\n"

    else:
        analysis += "- Could not extract numeric marks reliably\n"

    analysis += """
- GPT: semantic reasoning (baseline)
- RAG: evidence-grounded (retrieval-based)
"""

    return analysis