"""A gopherhole: a folder on disk, exposed as Gopher menus and documents."""

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = {".txt", ".text", ".md", ".markdown", ".rst", ".org", ".nfo", ".asc", ".csv", ".log", ""}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_TEXT_BYTES = 256_000
CACHE_DIR = ".gophergeist"


def item_type(path):
    if path.is_dir():
        return "1"
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return "0"
    if ext == ".gif":
        return "g"
    if ext in IMAGE_EXTS:
        return "I"
    return "9"


@dataclass
class Doc:
    selector: str
    path: Path
    text: str

    @property
    def title(self):
        return self.path.name


class Hole:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"{root} is not a folder")

    def selector_for(self, path):
        rel = path.relative_to(self.root).as_posix()
        return "/" if rel == "." else "/" + rel

    def resolve(self, selector):
        """Map a selector to a path inside the hole, or None. Hidden files and '..' never resolve."""
        parts = [p for p in selector.replace("\\", "/").split("/") if p]
        if any(p.startswith(".") for p in parts):
            return None
        path = self.root.joinpath(*parts).resolve() if parts else self.root
        try:
            path.relative_to(self.root)
        except ValueError:
            return None  # a symlink pointing outside the hole
        return path if path.exists() else None

    def listing(self, directory):
        children = [p for p in directory.iterdir() if not p.name.startswith(".")]
        return sorted(children, key=lambda p: (not p.is_dir(), p.name.lower()))

    def read_text(self, path):
        with open(path, "rb") as f:
            return f.read(MAX_TEXT_BYTES).decode("utf-8", errors="replace")

    def documents(self):
        """Every text document in the hole, depth first, skipping hidden things."""
        stack = [self.root]
        while stack:
            directory = stack.pop()
            for p in self.listing(directory):
                if p.is_dir():
                    stack.append(p)
                elif item_type(p) == "0":
                    yield Doc(self.selector_for(p), p, self.read_text(p))


def first_line(text, width=60):
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line if len(line) <= width else line[: width - 3].rstrip() + "..."
    return ""


SUMMARY_SYSTEM = (
    "You write one-line descriptions for entries in a Gopher menu. "
    "Reply with a single plain-text line of at most 60 characters describing what the document is about. "
    "No quotes, no markdown, no preamble."
)


class Summaries:
    """One-line model-written blurbs shown under each menu entry, cached by content hash."""

    def __init__(self, hole):
        self.hole = hole
        self.path = hole.root / CACHE_DIR / "summaries.json"
        self.lock = threading.Lock()
        try:
            self.cache = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            self.cache = {}

    @staticmethod
    def key(text):
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    def get(self, doc_or_path):
        if isinstance(doc_or_path, Doc):
            text = doc_or_path.text
        else:
            text = self.hole.read_text(doc_or_path)
        with self.lock:
            cached = self.cache.get(self.key(text))
        return cached or first_line(text)

    def missing(self):
        with self.lock:
            return [d for d in self.hole.documents() if self.key(d.text) not in self.cache]

    def build(self, backend, progress=None, before_each=None):
        """Summarize every new or changed document. Returns how many were written.
        before_each() is called before every model call; the server uses it to let visitors go first."""
        written = 0
        for doc in self.missing():
            if before_each:
                before_each()
            reply = backend.generate(SUMMARY_SYSTEM, f"File: {doc.title}\n\n{doc.text[:3000]}", max_tokens=40)
            line = first_line(reply.strip().strip('"'), width=60)
            if not line:
                continue
            with self.lock:
                self.cache[self.key(doc.text)] = line
            written += 1
            if progress:
                progress(doc, line)
            self.save()
        return written

    def save(self):
        with self.lock:
            data = json.dumps(self.cache, indent=1, sort_keys=True)
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(data)
