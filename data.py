#Step 1: load Simple English Wikipedia and split it into article -> paragraph -> sentence.



import gzip
import json
import os
import re

import requests

DATA_URL = "http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz"
DATA_PATH = "data/simplewiki-2020-11-01.jsonl.gz"

# A simple rule-based sentence splitter: split after . ! ? when followed by a space
# and a capital letter or digit. Good enough for Wikipedia prose, and easy to explain.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]


def download():
    if os.path.exists(DATA_PATH):
        return
    os.makedirs("data", exist_ok=True)
    print(f"Downloading {DATA_URL} ...")
    with requests.get(DATA_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def load_collection(max_articles=None):
    """Return (paragraphs, sentences).

    paragraphs: list of dicts {pid, title, article_id, pos, text}
                pos = position of the paragraph inside its article (used to add neighbours)
    sentences:  list of dicts {text, pid}  -> every sentence points back to its paragraph
    """
    download()
    paragraphs, sentences = [], []
    with gzip.open(DATA_PATH, "rt", encoding="utf8") as f:
        for n, line in enumerate(f):
            if max_articles and n >= max_articles:
                break
            page = json.loads(line)
            for pos, text in enumerate(page["paragraphs"]):
                pid = len(paragraphs)
                paragraphs.append(
                    {"pid": pid, "title": page["title"], "article_id": page["id"], "pos": pos, "text": text}
                )
                for s in split_sentences(text):
                    sentences.append({"text": s, "pid": pid})
    return paragraphs, sentences


if __name__ == "__main__":
    p, s = load_collection(max_articles=1000)
    print(f"{len(p)} paragraphs, {len(s)} sentences from 1000 articles")
    print(s[0])
