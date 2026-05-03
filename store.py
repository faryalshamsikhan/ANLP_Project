import faiss
import numpy as np
from rag.embedder import get_embedding
from ingestion.chunker import chunk_text


class RAGStore:

    def __init__(self):
        self.index = None
        self.data = []

    def build(self, dataset):

        # Step 1: dynamic embedding dimension
        sample_emb = get_embedding("test")
        dim = len(sample_emb)

        # Step 2: cosine similarity index
        self.index = faiss.IndexFlatIP(dim)

        for item in dataset:

            q = item["question"]
            full_text = q + "\n" + item["mark_scheme"]

            chunks = chunk_text(full_text)

            for chunk in chunks:

                emb = get_embedding(chunk)

                # normalize for cosine similarity
                emb = emb / np.linalg.norm(emb)

                self.index.add(np.array([emb], dtype="float32"))

                self.data.append({
                    "question": q,
                    "mark_scheme": item["mark_scheme"],
                    "chunk": chunk
                })

    def search(self, query, k=5):

        q_emb = get_embedding(query)
        q_emb = q_emb / np.linalg.norm(q_emb)

        scores, idx = self.index.search(
            np.array([q_emb], dtype="float32"),
            k
        )

        results = [self.data[i] for i in idx[0]]

        return results