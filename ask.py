"""Ask a question and see the answer, its sources, and the faithfulness report.

    python ask.py "Who invented the telephone?"
    python ask.py "Who invented the telephone?" --neighbours --dependence
"""
import argparse

from faithfulness import FaithfulnessChecker
from rag import RAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--neighbours", action="store_true", help="add paragraph before/after each hit")
    ap.add_argument("--dependence", action="store_true", help="run the leave-one-source-out test (slower)")
    args = ap.parse_args()

    rag = RAG()
    answer, sources = rag.answer(args.question, neighbours=args.neighbours)

    print("\nSOURCES")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s['title']}: {s['text'][:140]}...")
    print(f"\nANSWER\n  {answer}\n")

    checker = FaithfulnessChecker()
    print("SUPPORT CHECK (NLI)")
    for r in checker.support(answer, sources, rag=rag):
        if r["abstain"]:
            print(f"  [abstained] {r['text']}")
            continue
        flag = "OK " if r["supported"] else "UNSUPPORTED"
        cite = "citation ok" if r["citation_ok"] else f"cited {r['cites'] or 'nothing'}, best is [{r['best_source']}]"
        print(f"  {flag:11} p={r['p_entail']:.2f}  {cite:28} {r['text']}")

    if args.dependence:
        print("\nDEPENDENCE CHECK (remove one source, regenerate)")
        for d in checker.dependence(rag, args.question, sources, answer):
            print(f"  without [{d['removed']}]: {d['change']:.0%} of claims lost")
            for c in d["lost_claims"]:
                print(f"      lost: {c}")
            print(f"      new answer: {d['new_answer']}")


if __name__ == "__main__":
    main()
