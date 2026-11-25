import json
import math

import numpy as np
import ollama

# 1. Raw documents (our mini vector database)
documents = [
    "Remote employees receive a $500 annual stipend for home office equipment.",
    "Core working hours are 10:00 AM to 3:00 PM EST Monday through Friday.",
    "Unused vacation days roll over up to a maximum of 5 days per calendar year.",
]

# 2. Embed all documents into 1D vectors using local Ollama
doc_embeddings = [
    ollama.embed(model="nomic-embed-text", input=doc)["embeddings"][0]
    for doc in documents
]

# 3. User input query
for i in range(3):
    query = "How much money do I get for my home office?"
    query_embedding = ollama.embed(model="nomic-embed-text", input=query)["embeddings"][0]

    # 4. Retrieve the closest match using cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    best_idx = int(np.argmax(scores))
    retrieved_context = documents[best_idx]

    # 5. Call local qwen3.5:9b with the retrieved data injected into the prompt
    prompt = f"""Answer the question using ONLY the context below.

    Context:
    {retrieved_context}

    Question: {query}"""
    response = ollama.chat(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={"temperature": 0.1},
    )
    answer = response["message"]["content"]
    print(f"--- RETRIEVED DATA (Score: {scores[best_idx]:.4f}) ---")
    print(retrieved_context)
    print(answer)

    json_prompt = f"""Extract the monetary amount from the answer below.
    Return the amount as a number in the money field.
    If the answer does not contain a monetary amount, return null for money.

    Answer: {answer}"""
    response = ollama.chat(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": json_prompt}],
        format={
            "type": "object",
            "properties": {"money": {"type": ["number", "null"]}},
            "required": ["money"],
            "additionalProperties": False,
        },
        think=False,
        options={"temperature": 0.1},
    )
    result = json.loads(response["message"]["content"])
    if (
        not isinstance(result, dict)
        or set(result) != {"money"}
        or (
            result["money"] is not None
            and (
                isinstance(result["money"], bool)
                or not isinstance(result["money"], (int, float))
                or not math.isfinite(result["money"])
            )
        )
    ):
        raise ValueError("Ollama returned an invalid money response")
    print(json.dumps(result, allow_nan=False))
