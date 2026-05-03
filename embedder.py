from config import client, EMBED_MODEL

def get_embedding(texts):

    # texts = list[str]
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    return [d.embedding for d in response.data]