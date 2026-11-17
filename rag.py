#Step 3: retrieval-augmented generation with numbered sources.



import numpy as np
import requests
from sentence_transformers import SentenceTransformer

import index

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"

INSTRUCTION = (
    "You are a question answering bot. Answer the question using ONLY the sources below. "
    "After every sentence, cite the source it comes from, like [2]. "
    "If the sources do not contain the answer, say \"I don't know based on the sources.\" "
    "Keep the answer short (2-4 sentences)."
)


class RAG:
    def __init__(self):
        self.vecs, self.paragraphs, self.sentences = index.load()
        self.embedder = SentenceTransformer(index.EMBED_MODEL, device=index.device())
        # article_id -> {pos: pid}, used to find the paragraph before/after
        self._by_article = {}
        for p in self.paragraphs:
            self._by_article.setdefault(p["article_id"], {})[p["pos"]] = p["pid"]

    # ---------- retrieval ----------
    def retrieve(self, question, k_sentences=10, max_paragraphs=4, neighbours=False):
        q = self.embedder.encode(question, normalize_embeddings=True).astype(np.float16)
        scores = self.vecs @ q                      # cosine similarity to every sentence
        top = np.argsort(-scores)[: k_sentences]

        pids = []
        for i in top:                               # keep order of relevance, remove duplicates
            pid = self.sentences[i]["pid"]
            if pid not in pids:
                pids.append(pid)
        pids = pids[:max_paragraphs]

        sources = []
        for pid in pids:
            text = self.paragraphs[pid]["text"]
            if neighbours:                          # add the paragraph before and after
                text = self._with_neighbours(pid)
            sources.append({"pid": pid, "title": self.paragraphs[pid]["title"], "text": text})
        return sources

    def _with_neighbours(self, pid):
        p = self.paragraphs[pid]
        art = self._by_article[p["article_id"]]
        parts = [self.paragraphs[art[pos]]["text"] for pos in (p["pos"] - 1, p["pos"], p["pos"] + 1) if pos in art]
        return "\n".join(parts)

    # ---------- generation ----------
    @staticmethod
    def build_prompt(question, sources):
        ctx = "\n\n".join(f"[{i + 1}] ({s['title']}) {s['text']}" for i, s in enumerate(sources))
        return f"{INSTRUCTION}\n\nSources:\n{ctx}\n\nQuestion: {question}\nAnswer:"

    @staticmethod
    def generate(prompt):
        r = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "seed": 0}},   # deterministic -> repeatable experiments
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    def answer(self, question, **retrieve_kwargs):
        sources = self.retrieve(question, **retrieve_kwargs)
        return self.generate(self.build_prompt(question, sources)), sources
