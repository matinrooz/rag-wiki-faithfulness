"""Step 4: is the answer really grounded in the sources?

Two complementary checks, the same idea as my MSc thesis (do explanations match what the
model actually uses?), moved from ECG classifiers to an LLM with retrieved evidence:

"""
import re

import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"   # small (~70M params), runs on a laptop
ENTAIL_THRESHOLD = 0.5
_CITE = re.compile(r"\[(\d+)\]")


_ABBREV = {"st", "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "vs", "etc", "e.g", "i.e", "u.s", "no", "mt"}


def _rejoin_abbreviations(parts):
    """Undo splits after abbreviations like 'St.' or single initials like 'J.'"""
    out = []
    for p in parts:
        if out:
            last = out[-1].rstrip(".").split()[-1].lower() if out[-1].split() else ""
            if last in _ABBREV or (len(last) == 1 and last.isalpha()):
                out[-1] = out[-1] + " " + p
                continue
        out.append(p)
    return out


def split_answer(answer):
    """Split the answer into sentences and read the cited source numbers of each."""
    # LLMs often put citations after the full stop ("... balance. [1], [3]").
    # Move them in front of the full stop so they stay with their sentence.
    answer = re.sub(r"([.!?])\s*((?:\[\d+\][,\s]*)+)", lambda m: " " + m.group(2).strip(" ,") + m.group(1) + " ", answer)
    parts = _rejoin_abbreviations(re.split(r"(?<=[.!?])\s+", answer))
    out = []
    for s in parts:
        s = s.strip()
        if not re.search(r"[A-Za-z]{3}", _CITE.sub("", s)):
            continue
        cites = [int(c) for c in _CITE.findall(s)]
        text = re.sub(r"[\s,]+([.!?])$", r"\1", _CITE.sub("", s)).strip()
        out.append({"text": text, "cites": cites})
    return out


DECOMPOSE_PROMPT = (
    "Break the sentence into short, independent facts. Each fact must be a full sentence "
    "that names its subject (no 'it', 'they'). Output one fact per line, nothing else.\n\n"
    "Sentence: {s}\nFacts:"
)


def decompose(rag, sentence):
    """Split one answer sentence into atomic claims with the local LLM (FActScore-style).

    NLI models often say 'neutral' when one hypothesis packs two facts, even if the source
    states both (e.g. 'X is a town in Bavaria and the capital of Upper Franconia').
    """
    out = rag.generate(DECOMPOSE_PROMPT.format(s=sentence))
    facts = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in out.splitlines()]
    facts = [f for f in facts if len(f) > 8]
    return facts or [sentence]


class FaithfulnessChecker:
    def __init__(self):
        self.nli = CrossEncoder(NLI_MODEL)
        # label order of this model: contradiction, entailment, neutral
        labels = [l.lower() for l in self.nli.model.config.id2label.values()]
        self.entail_idx = labels.index("entailment")

    def _nli(self, pairs):
        logits = self.nli.predict(pairs, convert_to_numpy=True)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, self.entail_idx]

    def entail_probs(self, premises, hypothesis, titles=None):
        """P(premise entails hypothesis) for each premise.

        NLI models are trained on single-sentence premises and fail on whole paragraphs
        (tested: 0.02 on the full Bayreuth paragraph vs 0.99 on its first sentence).
        So, like SummaC (Laban et al., 2022), each premise is cut into sentences and pairs
        of neighbouring sentences (pairs keep 'It is the capital...' linked to its subject),
        and the premise score is the max over these units.
        """
        units, owner = [], []
        for i, p in enumerate(premises):
            sents = _rejoin_abbreviations(re.split(r"(?<=[.!?])\s+", p)) or [p]
            if titles:
                # Simple coreference: Wikipedia sentences like "It can be used to make glass."
                # refer to the article title, often many sentences back.
                sents = [re.sub(r"^(It|They|This)\b", titles[i], s) for s in sents]
            for j in range(len(sents)):
                units.append(sents[j]); owner.append(i)
                if j + 1 < len(sents):
                    units.append(sents[j] + " " + sents[j + 1]); owner.append(i)
        probs = self._nli([(u, hypothesis) for u in units])
        out = np.zeros(len(premises))
        for o, pr in zip(owner, probs):
            out[o] = max(out[o], pr)
        return out

    def support(self, answer, sources, rag=None):
        """Per answer sentence: is it entailed by the sources, and by the source it cites?

        If `rag` is given, each sentence is first split into atomic claims; the sentence is
        supported only if EVERY claim is entailed by some source (score = weakest claim).
        """
        premises = [s["text"] for s in sources]
        titles = [s["title"] for s in sources]
        rows = []
        for sent in split_answer(answer):
            if "don't know" in sent["text"].lower():
                rows.append({**sent, "abstain": True})
                continue
            claims = [sent["text"]]
            if rag:
                # Guard: a small LLM can change the meaning while splitting (seen in testing:
                # 'X is famous for a car factory' -> 'X is not famous for a car factory').
                # Keep only claims that the original sentence itself entails.
                atoms = decompose(rag, sent["text"])
                keep = self._nli([(sent["text"], a) for a in atoms]) >= ENTAIL_THRESHOLD
                claims = [a for a, k in zip(atoms, keep) if k] or claims
            # probs[c, s] = P(source s entails claim c)
            probs = np.stack([self.entail_probs(premises, c, titles) for c in claims])
            claim_best = probs.max(axis=1)                     # best source for each claim
            entailing = {i + 1 for i in range(len(sources)) if (probs[:, i] >= ENTAIL_THRESHOLD).any()}
            cited_ok = [c for c in sent["cites"] if 1 <= c <= len(sources)]
            rows.append({
                **sent,
                "abstain": False,
                "claims": [{"text": c, "p_entail": float(p)} for c, p in zip(claims, claim_best)],
                "best_source": int(np.argmax(probs.max(axis=0))) + 1,
                "p_entail": float(claim_best.min()),           # weakest claim decides
                "supported": bool((claim_best >= ENTAIL_THRESHOLD).all()),
                "citation_ok": bool(cited_ok) and all(
                    (probs[:, [c - 1 for c in cited_ok]].max(axis=1) >= ENTAIL_THRESHOLD)
                ),
            })
        return rows

    def dependence(self, rag, question, sources, answer):
        """Leave-one-source-out: which claims of the answer disappear without each source?

        For every claim (sentence) of the original answer we ask the NLI model whether the
        NEW answer still entails it. change = fraction of original claims that are lost.
        (A first version compared whole-answer embeddings; that was too coarse: every
        removal gave ~0.13 even when the content changed a lot.)
        """
        claims = [c["text"] for c in split_answer(answer) if "don't know" not in c["text"].lower()]
        out = []
        for i in range(len(sources)):
            reduced = sources[:i] + sources[i + 1:]
            new = rag.generate(rag.build_prompt(question, reduced))
            new_plain = _CITE.sub("", new)
            kept = [float(self.entail_probs([new_plain], c)[0]) >= ENTAIL_THRESHOLD for c in claims]
            lost = [c for c, k in zip(claims, kept) if not k]
            out.append({
                "removed": i + 1,
                "change": len(lost) / max(len(claims), 1),
                "lost_claims": lost,
                "new_answer": new,
            })
        return out
