"""
Retrieval layer.

Design choice: TF-IDF + cosine similarity over the chunk corpus, not a
dense-embedding vector DB. Rationale (see README trade-offs section):
  - The corpus is tiny (40 chunks). Dense embeddings buy semantic recall on
    large/paraphrase-heavy corpora; on 40 short, keyword-rich support
    documents TF-IDF is competitive and fully deterministic.
  - Zero external dependency: no embedding API call, no model download, no
    network access needed at retrieval time -> the retrieval half of the
    pipeline is testable offline and has zero marginal cost per query.
  - The interface (`Retriever.search`) is the seam: swapping in a real
    embedding index (e.g. Gemini `text-embedding-004` + a vector DB) later
    only touches this file.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.jsonl"


class Retriever:
    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        corpus = [f"{c['title']}\n{c['text']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 4):
        """Return top_k chunks with similarity scores, sorted descending."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(sims[idx])})
        return results


if __name__ == "__main__":
    r = Retriever()
    for q in [
        "How long do I have to get a refund on a course?",
        "my certificate never showed up",
        "can I download videos to watch on a plane",
    ]:
        print("\nQ:", q)
        for hit in r.search(q, top_k=3):
            print(f"  [{hit['score']:.3f}] {hit['id']} — {hit['title']}")
