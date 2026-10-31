"""Veronica, raised from the dead.

The original Veronica (1992, University of Nevada, Reno) was Gopherspace's search engine:
you typed words into a type 7 menu item and got back a menu of matching items. This one
keeps that shape, but a local open-weight model reads the best matches and answers you,
citing the documents it used. Retrieval is a plain BM25 index, so search still works with
no model at all.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOPWORDS = set(
    "a an and are as at be but by can did do does for from had has have how i if in into is it its "
    "me my of on or so than that the their them then there these they this to was we were what when "
    "where which who why will with you your about".split()
)
CHUNK_CHARS = 1500

# Speed on a CPU is mostly prompt length and answer length, so both are kept small:
# a short system prompt, three focused passages, and answers capped at about 80 words.
SYSTEM = (
    "You are Veronica, the ghost of Gopher's 1992 search engine. Answer ONLY from the numbered sources, "
    "citing them like [1]. If they don't say, reply that the archives are silent. Plain text, under 80 "
    "words, at most one short ghostly aside."
)
TOP_K = 3
PASSAGE_CHARS = 450
MAX_TOKENS = 160


def stem(word):
    """A deliberately crude stemmer: enough that 'licensing' finds 'license' and 'crawled' finds 'crawl'."""
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


@dataclass
class Chunk:
    selector: str
    title: str
    text: str
    terms: Counter = field(default_factory=Counter)
    title_terms: frozenset = frozenset()


@dataclass
class Answer:
    text: str
    sources: list  # [(selector, title)], numbered in order from 1
    spoke: bool  # True if a model wrote the text, False for the plain-index fallback


def chunk_text(text, size=CHUNK_CHARS):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


class Index:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.chunks = []
        for doc in docs:
            for piece in chunk_text(doc.text):
                title_terms = frozenset(tokenize(doc.title.rsplit(".", 1)[0]))
                self.chunks.append(Chunk(doc.selector, doc.title, piece, Counter(tokenize(piece)), title_terms))
        self.df = Counter()
        for c in self.chunks:
            self.df.update(c.terms.keys())
        lengths = [sum(c.terms.values()) for c in self.chunks]
        self.avg_len = (sum(lengths) / len(lengths)) if lengths else 1

    def search(self, query, k=4):
        q = set(tokenize(query))
        n = len(self.chunks)
        scored = []
        for c in self.chunks:
            length = sum(c.terms.values()) or 1
            score = 0.0
            for t in q:
                tf = c.terms.get(t, 0)
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                if t in c.title_terms:
                    score += 1.5 * idf  # a file named archie.txt is about Archie
                if not tf:
                    continue
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda pair: -pair[0])
        return [c for _, c in scored[:k]]


def _sources(chunks):
    sources = []
    for c in chunks:
        if (c.selector, c.title) not in sources:
            sources.append((c.selector, c.title))
    return sources


def best_sentence(text, query, limit=240):
    """The sentence of a passage that shares the most words with the query."""
    q = set(tokenize(query))
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        sentences += re.split(r"(?<=[.!?])\s+", " ".join(para.split()))
    # Most shared words wins; on a tie, a real sentence beats a bare heading.
    best = max(sentences, key=lambda s: (len(q & set(tokenize(s))), len(s) > 40))
    return best if len(best) <= limit else best[: limit - 3].rstrip() + "..."


def _fallback(query, chunks, sources, note):
    lines, seen = [note, ""], set()
    for c in chunks:
        if c.selector in seen:
            continue  # one line per document reads better than three from the same file
        seen.add(c.selector)
        n = sources.index((c.selector, c.title)) + 1
        lines.append(f"[{n}] {best_sentence(c.text, query)}")
    return "\n".join(lines)


def focus(text, query, limit=PASSAGE_CHARS):
    """The few sentences of a passage most related to the query, kept in their original order.
    Feeding the model 450 relevant characters instead of 1,500 makes CPU answers much quicker."""
    if len(text) <= limit:
        return text
    q = set(tokenize(query))
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        sentences += re.split(r"(?<=[.!?])\s+", " ".join(para.split()))
    ranked = sorted(range(len(sentences)), key=lambda i: -len(q & set(tokenize(sentences[i]))))
    keep, used = set(), 0
    for i in ranked:
        if used + len(sentences[i]) > limit and keep:
            continue
        keep.add(i)
        used += len(sentences[i]) + 1
    return " ".join(sentences[i] for i in sorted(keep))


def prepare(query, index):
    """Retrieve and build the prompt. Returns (chunks, sources, prompt), or None if nothing matches."""
    chunks = index.search(query, k=TOP_K)
    if not chunks:
        return None
    sources = _sources(chunks)
    numbered = []
    for c in chunks:
        n = sources.index((c.selector, c.title)) + 1
        numbered.append(f"[{n}] {c.title}: {focus(c.text, query)}")
    prompt = "Sources:\n" + "\n".join(numbered) + f"\n\nQuestion: {query}"
    return chunks, sources, prompt


SILENT = Answer("The archives are silent. Nothing in this hole speaks of that.", [], spoke=False)


def ask(query, index, backend=None):
    plan = prepare(query, index)
    if plan is None:
        return SILENT
    chunks, sources, prompt = plan
    if backend is None:
        return Answer(_fallback(query, chunks, sources, "No spirit is bound to this server, so the index speaks alone:"),
                      sources, spoke=False)
    try:
        text = backend.generate(SYSTEM, prompt, max_tokens=MAX_TOKENS)
    except Exception as e:  # a sleeping model must never take the server down with it
        return Answer(_fallback(query, chunks, sources, f"The spirit would not wake ({e}). The index speaks instead:"),
                      sources, spoke=False)
    if not text.strip():
        return Answer(_fallback(query, chunks, sources, "The spirit said nothing. The index speaks instead:"),
                      sources, spoke=False)
    return Answer(text.strip(), sources, spoke=True)


def ask_stream(query, index, backend=None):
    """Like ask(), but returns (sources, pieces) where pieces yields the answer as the model writes it,
    so a visitor sees the first words in a second or two instead of waiting for the whole reply.
    Backends without streaming, or no backend, yield the whole answer as one piece."""
    if backend is None or not hasattr(backend, "stream"):
        answer = ask(query, index, backend)
        return answer.sources, iter([answer.text])
    plan = prepare(query, index)
    if plan is None:
        return [], iter([SILENT.text])
    chunks, sources, prompt = plan

    def pieces():
        said = False
        try:
            for piece in backend.stream(SYSTEM, prompt, max_tokens=MAX_TOKENS):
                said = said or bool(piece.strip())
                yield piece
        except Exception as e:
            if said:
                yield " ... (the spirit faded mid-sentence)"
            else:
                yield _fallback(query, chunks, sources, f"The spirit would not wake ({e}). The index speaks instead:")
            return
        if not said:
            yield _fallback(query, chunks, sources, "The spirit said nothing. The index speaks instead:")

    return sources, pieces()
