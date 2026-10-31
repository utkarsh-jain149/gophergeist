"""Build a fine-tuning dataset that teaches a small model to *be* Veronica.

A bigger open "teacher" model (say qwen2.5:7b on a Colab GPU) writes the questions and the answers;
a 1B "student" is then fine-tuned on them. The student learns behaviour, not facts: answer only from
the numbered sources, cite them, stay short, admit when the archives are silent. Facts still come
from retrieval at runtime, so editing the hole never needs retraining.

Every example uses exactly the prompt the server builds at runtime (veronica.prepare), so the
student is trained on what it will really see.
"""

import json
import random
import re

from .veronica import SYSTEM, Index, chunk_text, prepare

QUESTION_SYSTEM = (
    "You write questions that a curious visitor to an old internet archive might type into a search box. "
    "Each question must be answerable from the passage alone. Mix short keyword-style queries with full "
    "questions. One question per line, no numbering, no other text."
)

# The teacher gets a richer brief than the student's one-line runtime prompt, so its answers show
# the behaviour we want; the student then learns to produce them from the short prompt alone.
TEACHER_SYSTEM = (
    "You are Veronica, the ghost of Gopherspace's 1992 search engine, answering from an old server. "
    "Rules, in order of importance:\n"
    "1. Use ONLY facts stated in the numbered sources. Never add outside knowledge, even if you know it.\n"
    "2. Cite the source of every fact inline like [1] or [2].\n"
    "3. If the sources do not answer the question, reply with one or two sentences saying the archives "
    "are silent on that, and cite nothing.\n"
    "4. Plain text, 25 to 70 words. No markdown, no lists, no preamble like 'Based on the sources'.\n"
    "5. You may end with one short, dry, ghostly aside (under 12 words). Accuracy matters more than mood."
)

# Questions this kind of hole never answers. Retrieval sometimes still finds loosely related
# passages for them, which is exactly when a small model is tempted to make something up.
OFF_TOPIC = [
    "What is the best pizza topping?", "Who won the 2018 World Cup?", "How do I bake sourdough bread?",
    "What is the capital of Australia?", "How do I fix a flat bicycle tire?", "What's the weather tomorrow?",
    "Who painted the Mona Lisa?", "How many moons does Jupiter have?", "What is the price of bitcoin?",
    "How do I center a div in CSS?", "Who wrote Pride and Prejudice?", "What is a black hole?",
    "How do I train a puppy?", "What year did the Titanic sink?", "What is the speed of light?",
]


def parse_questions(text, limit):
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if 8 <= len(line) <= 160 and line not in out:
            out.append(line)
    return out[:limit]


def good_answer(text, answerable):
    words = len(text.split())
    if not 5 <= words <= 110 or "```" in text or text.lstrip().startswith(("#", "*", "-")):
        return False
    if answerable:
        return bool(re.search(r"\[\d\]", text))
    return "silent" in text.lower() and not re.search(r"\[\d\]", text)


def example(prompt, answer):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer.strip()},
    ]}


def build(docs, teacher, per_passage=3, mismatched=0.15, seed=7, progress=None):
    """Return a list of chat examples. teacher is any backend with generate(system, prompt, max_tokens)."""
    rng = random.Random(seed)
    docs = list(docs)
    index = Index(docs)
    questions = []  # (question, answerable)
    for doc in docs:
        for passage in chunk_text(doc.text):
            reply = teacher.generate(QUESTION_SYSTEM, f"Passage from {doc.title}:\n\n{passage}\n\n"
                                     f"Write {per_passage} questions.", max_tokens=200)
            for q in parse_questions(reply, per_passage):
                questions.append((q, True))
            if progress:
                progress(f"questions from {doc.selector}")
    questions += [(q, False) for q in OFF_TOPIC]

    examples = []
    for q, answerable in questions:
        plan = prepare(q, index)
        if plan is None:
            continue  # the server answers these without the model, so nothing to learn
        _, sources, prompt = plan
        if answerable and rng.random() < mismatched:
            # Swap in sources from unrelated documents: the right answer becomes "the archives are silent".
            others = [d for d in docs if d.selector not in {s for s, _ in sources}]
            if others:
                alien = Index(rng.sample(others, min(2, len(others))))
                plan = prepare(" ".join(rng.choice(alien.chunks).text.split()[:12]), alien)
                if plan:
                    prompt = plan[2].rsplit("\n\nQuestion: ", 1)[0] + f"\n\nQuestion: {q}"
                    answerable = False
        answer = teacher.generate(TEACHER_SYSTEM, prompt, max_tokens=200).strip()
        if good_answer(answer, answerable):
            examples.append(example(prompt, answer))
        if progress:
            progress(f"{'answered' if answerable else 'refused'}: {q}")
    rng.shuffle(examples)
    return examples


def write_jsonl(examples, train_path, eval_path=None, eval_share=0.1):
    n_eval = int(len(examples) * eval_share) if eval_path else 0
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in examples[n_eval:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    if eval_path:
        with open(eval_path, "w", encoding="utf-8") as f:
            for ex in examples[:n_eval]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples) - n_eval, n_eval
