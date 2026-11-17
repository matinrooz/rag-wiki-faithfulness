"""Run the faithfulness checks on a small question set and report aggregate numbers.

    

Main question: do the LLM's citations match the sources it actually depends on?
We compare how many claims are lost when we remove a CITED source vs an UNCITED source.
If citations were faithful, removing a cited source should hurt much more.
"""
import json
import os

import numpy as np

from faithfulness import FaithfulnessChecker, split_answer
from rag import RAG

# Questions about articles that are inside the 20k-article development subset.
QUESTIONS = [
    "What does the cerebellum do?",
    "Where is the Sistine Chapel and who painted its ceiling?",
    "What is lithium carbonate used for?",
    "What is a pulley?",
    "What is Remembrance Day?",
    "Where does the harbor seal live?",
    "What are amino acids?",
    "What is a prognosis in medicine?",
    "Where is Bayreuth?",
    "What does a playwright do?",
]


def main():
    rag, checker = RAG(), FaithfulnessChecker()
    records = []
    for q in QUESTIONS:
        answer, sources = rag.answer(q)
        support = checker.support(answer, sources, rag=rag)
        dep = checker.dependence(rag, q, sources, answer)
        cited = {c for s in split_answer(answer) for c in s["cites"]}
        records.append({"question": q, "answer": answer,
                        "sources": [s["title"] for s in sources],
                        "support": support, "dependence": dep, "cited": sorted(cited)})
        print(f"done: {q}")

    claims = [r for rec in records for r in rec["support"] if not r["abstain"]]
    lost_cited = [d["change"] for rec in records for d in rec["dependence"] if d["removed"] in rec["cited"]]
    lost_uncited = [d["change"] for rec in records for d in rec["dependence"] if d["removed"] not in rec["cited"]]
    decorative = [d for rec in records for d in rec["dependence"] if d["removed"] in rec["cited"] and d["change"] == 0]

    summary = {
        "questions": len(records),
        "answer_sentences": len(claims),
        "supported_by_some_source": float(np.mean([c["supported"] for c in claims])),
        "sentences_with_a_citation": float(np.mean([bool(c["cites"]) for c in claims])),
        "citation_points_to_entailing_source": float(np.mean([c["citation_ok"] for c in claims])),
        "claims_lost_removing_cited_source": float(np.mean(lost_cited)) if lost_cited else None,
        "claims_lost_removing_uncited_source": float(np.mean(lost_uncited)) if lost_uncited else None,
        "cited_sources_whose_removal_changes_nothing": f"{len(decorative)}/{len(lost_cited)}",
    }
    os.makedirs("results", exist_ok=True)
    with open("results/eval.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
