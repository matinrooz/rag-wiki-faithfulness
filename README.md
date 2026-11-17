# rag-wiki-faithfulness

RAG over Simple English Wikipedia, plus two checks on the answers:

- **support**: is each claim in the answer actually written in one of the retrieved sources?
- **dependence**: if I remove a source and ask again, does the answer change?

In my MSc thesis I tested if attribution methods for ECG classifiers (IG, SmoothGrad-IG, DTD)
point to the parts of the signal the model really uses, by perturbing those parts. This
project is the same question for an LLM. The citation [n] is the explanation, and removing
source n is the perturbation.

Everything runs on my laptop (MacBook Air, 8 GB):
- retrieval: `all-MiniLM-L6-v2`
- LLM: `qwen2.5:3b` with Ollama
- NLI: `cross-encoder/nli-deberta-v3-xsmall`

## How it works

1. The question is compared with all sentences (sentence embeddings, dot product).
2. The paragraphs of the 10 best sentences are collected, duplicates removed, max 4.
3. The LLM gets them as sources [1]-[4] and has to cite after every sentence.
4. Each answer sentence is split into small claims, and an NLI model checks which source
   entails each claim.
5. Each source is removed one at a time, the answer is generated again, and I check which
   of the original claims are gone.

## Example

Output of `python ask.py "Where is the Sistine Chapel and who painted its ceiling?" --dependence`
(source texts shortened):

```
SOURCES
  [1] Chapel: Some of the most famous large private chapels are St. George's Chapel at Windsor ...
  [2] Michelangelo: The main reason that Pope Julius' tomb was not finished was that ...
  [3] Sistine Chapel: The ceiling is the most famous part of the Sistine Chapel, and ...
  [4] Paint: Paintings are pictures that are done in paint. ...

ANSWER
  The Sistine Chapel is near St. Peter's Basilica. Michelangelo painted its ceiling. [2], [1]

SUPPORT CHECK (NLI)
  OK   p=0.87  cited nothing, best is [2]   The Sistine Chapel is near St. Peter's Basilica.
  OK   p=1.00  citation ok                  Michelangelo painted its ceiling.

DEPENDENCE CHECK (remove one source, regenerate)
  without [1]: 0% of claims lost
  without [2]: 50% of claims lost
      lost: The Sistine Chapel is near St. Peter's Basilica.
  without [3]: 0% of claims lost
  without [4]: 0% of claims lost
```

Both sentences are supported, but the model cited [1] as well, and removing [1] changes
nothing. Only [2] matters.

## Results

10 questions, using the first 20k articles only.

| | |
|---|---|
| answer sentences supported by a source | 14 / 16 |
| sentences with a citation | 10 / 16 |
| cited sentences where the cited source really supports it | 9 / 10 |
| claims lost when removing a cited source | 67% |
| claims lost when removing an uncited source | 18% |
| cited sources where removing them changed nothing | 3 / 15 |

So most citations mean something, but not all. Three examples where the citation looked fine
but the answer did not depend on it:

- Sistine Chapel: it cited "Chapel" (a list of famous chapels) next to "Michelangelo painted
  its ceiling". Removing it changed nothing, removing the Michelangelo article did.
- Prognosis: it cited the *Wilms tumor* article, but the definition came from another source.
- Cerebellum: after I removed the right source, the model cited *Goldfish* and wrote that
  the cerebellum "controls internal organ functions", which is not in any source.

Full outputs are in `results/eval.json`. The two older files in `results/` are from earlier
versions of the checker.

## Problems I ran into

- First I measured dependence by comparing embeddings of the whole answer. Every removal
  gave about 0.13, even when the answer clearly changed. Now I check claim by claim with NLI.
- The NLI model is bad with long premises. "Bayreuth is a town in northern Bavaria,
  Germany." got 0.02 against the full paragraph and 0.99 against its first sentence. The base
  model did the same. So now sources are checked per sentence and per pair of sentences
  (the same idea as SummaC).
- Sentences with two facts were often "neutral". I let the LLM split them into single
  claims first, like in FActScore.
- The 3B model sometimes changes the meaning when splitting. Once "Bayreuth is famous for
  its large car factory" became "Bayreuth is not famous for a car factory". I only keep a
  claim if the original sentence entails it.
- "It can be used to make glass" was not matched to *Lithium carbonate* because the name is
  four sentences earlier. I replace a leading It/They/This with the article title.

Supported sentences went from 12/16 to 14/16 after these fixes.

## Limitations

10 questions is not a benchmark, it is more a demo of the method. The NLI model still misses
some cases (time and event three sentences apart). The dependence test only sees what the
answer needs from the sources. If the model knows something by itself (Michelangelo), it can
still answer without the source. And qwen sometimes just cites [1][2][3][4] at once.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install ollama && brew services start ollama && ollama pull qwen2.5:3b

.venv/bin/python index.py --max-articles 20000    # about 15 min on my laptop
.venv/bin/python ask.py "Where is Bayreuth?" --dependence
.venv/bin/python evaluate.py
```

Files: `data.py` (load and split), `index.py` (embeddings), `rag.py` (retrieval + LLM),
`faithfulness.py` (both checks), `ask.py` (one question), `evaluate.py` (10 questions).

## References

- Laban et al. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in
  Summarization. TACL. [arXiv:2111.09525](https://arxiv.org/abs/2111.09525)
- Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long
  Form Text Generation. EMNLP. [arXiv:2305.14251](https://arxiv.org/abs/2305.14251)
- Dataset: [Simple English Wikipedia paragraphs](http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz)
  from the sentence-transformers project
- Models: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  [nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall),
  [Qwen2.5-3B](https://ollama.com/library/qwen2.5) via [Ollama](https://ollama.com)

## Author

[Matin Roozbahani](https://github.com/matinrooz), MSc Computer Science (Machine
Intelligence), University of Basel
