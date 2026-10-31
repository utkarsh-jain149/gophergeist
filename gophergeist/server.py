"""The Gopher server: asyncio, one request line in, one response out, connection closed. Just like 1991."""

import asyncio
import contextlib
import textwrap
import threading
from collections import OrderedDict

from . import __version__, protocol
from .hole import Hole, Summaries, item_type
from .veronica import Answer, Index, ask, ask_stream

SEARCH = "/~veronica"
ABOUT = "/~about"
WIDTH = 67

BANNER = r"""
   .-.
  (o o)   g o p h e r g e i s t
  | O \   a gopherhole, haunted by an open-weight model
   \   \
    `~~~'
""".strip("\n").splitlines()

ABOUT_TEXT = """\
About this ghost
================

This is a Gopher server (RFC 1436, 1993). Gopher was the friendly,
menu-driven internet that the web replaced. A big reason it faded:
in 1993 the University of Minnesota announced license fees for its
server, and that same spring CERN put the web in the public domain.

The search item on the main menu revives Veronica, Gopherspace's
search engine from 1992. A plain BM25 index finds the passages, and
an open-weight model running on this machine reads them and answers,
citing its sources.

Nothing you ask leaves this machine. No API key. No account. If the
model is asleep, the index still answers on its own.

Served by gophergeist {version}. Spirit: {spirit}.
"""


class GopherServer:
    def __init__(self, root, backend=None, hostname="localhost", port=7070, summaries=True):
        self.hole = Hole(root)
        self.backend = backend
        self.hostname = hostname
        self.port = port
        self.summaries = Summaries(self.hole) if summaries else None
        self._index = None
        self._index_sig = None
        self._index_lock = threading.Lock()
        # Visitors ask the same few questions, so remember recent answers. The key includes the
        # index signature, so editing a document makes old answers stale automatically.
        self._answers = OrderedDict()
        self._answers_lock = threading.Lock()
        # Ollama answers one request at a time by default, so background menu blurbs must never make
        # a visitor's question wait in line behind them. Blurbs pause while any question is running.
        self._asking = 0
        self._idle = threading.Condition()

    # ---- menus -------------------------------------------------------

    def _link(self, itype, display, selector):
        return protocol.item(itype, display, selector, self.hostname, self.port)

    def _wrap_info(self, text, indent=""):
        lines = []
        for para in text.splitlines() or [""]:
            wrapped = textwrap.wrap(para, WIDTH - len(indent)) or [""]
            lines += [protocol.info(indent + w) for w in wrapped]
        return lines

    def menu(self, directory):
        lines = []
        at_root = directory == self.hole.root
        if at_root:
            lines += [protocol.info(b) for b in BANNER]
            lines.append(protocol.info())
            lines.append(self._link("7", "Ask Veronica (search, answered by a local ghost)", SEARCH))
            lines.append(protocol.info())
        else:
            lines.append(protocol.info(f"--- {self.hole.selector_for(directory)} ---"))
            lines.append(protocol.info())
        for path in self.hole.listing(directory):
            itype = item_type(path)
            selector = self.hole.selector_for(path)
            lines.append(self._link(itype, path.name + ("/" if itype == "1" else ""), selector))
            if itype == "0" and self.summaries:
                blurb = self.summaries.get(path)
                if blurb:
                    lines.append(protocol.info("    " + blurb))
        if at_root:
            lines.append(protocol.info())
            lines.append(self._link("0", "About this ghost", ABOUT))
        return protocol.end_menu(lines)

    # ---- search ------------------------------------------------------

    def index(self):
        """The BM25 index, rebuilt whenever a document in the hole changes."""
        with self._index_lock:
            docs = list(self.hole.documents())
            sig = tuple((d.selector, len(d.text), hash(d.text)) for d in docs)
            if sig != self._index_sig:
                self._index, self._index_sig = Index(docs), sig
            return self._index

    def search(self, query):
        if not query:
            return protocol.end_menu([protocol.error("Veronica needs a question. Type some words to search.")])
        answer = self.cached_answer(query)
        if answer is None:
            with self.question():
                answer = self.remember(query, ask(query, self.index(), self.backend))
        lines = [protocol.info(f'Veronica, asked: "{query}"'), protocol.info()]
        lines += self._wrap_info(answer.text)
        if answer.sources:
            lines += [protocol.info(), protocol.info("Sources:")]
            for n, (selector, title) in enumerate(answer.sources, 1):
                lines.append(self._link("0", f"[{n}] {title}", selector))
        lines += [protocol.info(), self._link("7", "Ask another question", SEARCH),
                  self._link("1", "Back to the main menu", "/")]
        return protocol.end_menu(lines)

    # ---- dispatch ----------------------------------------------------

    def respond(self, selector, query=""):
        if selector in ("", "/"):
            return self.menu(self.hole.root)
        if selector == SEARCH:
            return self.search(query)
        if selector == ABOUT:
            return protocol.text_body(ABOUT_TEXT.format(version=__version__, spirit=self.backend or "none (index only)"))
        path = self.hole.resolve(selector)
        if path is None:
            return protocol.end_menu([protocol.error(f"Nothing lives at {selector} anymore. Only the wind.")])
        if path.is_dir():
            return self.menu(path)
        if item_type(path) == "0":
            return protocol.text_body(self.hole.read_text(path))
        return path.read_bytes()  # binary types are sent raw; the closed connection marks the end

    async def handle(self, reader, writer):
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            selector, query = protocol.parse_request(line)
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, self.respond, selector, query)
            writer.write(data)
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def start(self, bind="127.0.0.1"):
        server = await asyncio.start_server(self.handle, bind, self.port)
        if self.port == 0:  # tests ask for any free port
            self.port = server.sockets[0].getsockname()[1]
        return server

    @contextlib.contextmanager
    def question(self):
        with self._idle:
            self._asking += 1
        try:
            yield
        finally:
            with self._idle:
                self._asking -= 1
                self._idle.notify_all()

    def wait_until_idle(self):
        with self._idle:
            self._idle.wait_for(lambda: self._asking == 0)

    def _answer_key(self, query):
        self.index()  # refresh the signature if documents changed
        return (" ".join(query.lower().split()), self._index_sig)

    def cached_answer(self, query):
        key = self._answer_key(query)
        with self._answers_lock:
            if key in self._answers:
                self._answers.move_to_end(key)
                return self._answers[key]
        return None

    def remember(self, query, answer, limit=256):
        if answer.spoke:  # only cache real answers, never a fallback from a sleeping model
            with self._answers_lock:
                self._answers[self._answer_key(query)] = answer
                while len(self._answers) > limit:
                    self._answers.popitem(last=False)
        return answer

    def answer_stream(self, query):
        """(sources, pieces) for the web view. Cached answers come back whole and instantly."""
        cached = self.cached_answer(query)
        if cached:
            return cached.sources, iter([cached.text])
        sources, pieces = ask_stream(query, self.index(), self.backend)
        if self.backend is None or not hasattr(self.backend, "stream") or not sources:
            return sources, pieces

        def remembering():
            text = []
            with self.question():
                for piece in pieces:
                    text.append(piece)
                    yield piece
            full = "".join(text).strip()
            if full and "The index speaks instead" not in full and not full.endswith("mid-sentence)"):
                self.remember(query, Answer(full, sources, spoke=True))

        return sources, remembering()

    def warm_in_background(self, progress=None):
        """Load the model before the first visitor arrives."""
        if not (self.backend and hasattr(self.backend, "warm")):
            return None

        def run():
            try:
                self.backend.warm()
                if progress:
                    progress(None, "spirit is awake and waiting")
            except Exception as e:
                if progress:
                    progress(None, f"warm-up failed: {e}")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def summarize_in_background(self, progress=None):
        if not (self.backend and self.summaries):
            return None

        def run():
            try:
                self.summaries.build(self.backend, progress, before_each=self.wait_until_idle)
            except Exception as e:
                if progress:
                    progress(None, f"summaries stopped: {e}")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
