"""Tests run fully offline. A fake 'spirit' stands in for the model, and a stub HTTP server
stands in for Ollama / llama.cpp, so no weights are needed."""

import asyncio
import json
import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from gophergeist import client, protocol
from gophergeist.hole import Hole, Summaries
from gophergeist.model import ModelError, Ollama, OpenAICompatible, make_backend
from gophergeist.server import SEARCH, GopherServer
from gophergeist.veronica import Index, ask, chunk_text, stem

DEMO = Path(__file__).resolve().parent.parent / "demo" / "hole"


class FakeSpirit:
    """Records what it was asked and answers with a fixed reply."""

    def __init__(self, reply="Gopher faded after the 1993 license fees [1]. The wind remembers."):
        self.reply = reply
        self.calls = []

    def __str__(self):
        return "fake-spirit"

    def generate(self, system, prompt, max_tokens=300):
        self.calls.append((system, prompt))
        return self.reply


class SleepingSpirit:
    def generate(self, system, prompt, max_tokens=300):
        raise ModelError("Could not reach http://localhost:11434")


@pytest.fixture
def hole(tmp_path):
    root = tmp_path / "hole"
    shutil.copytree(DEMO, root)
    return root


# ---- protocol ----------------------------------------------------------

def test_menu_line_strips_tabs_and_newlines():
    line = protocol.item("0", "evil\tname\n", "/a\tb", "localhost", 70)
    assert line == "0evil name \t/a b\tlocalhost\t70\r\n"


def test_text_body_dot_stuffs_and_terminates():
    body = protocol.text_body("hello\n.hidden line\n.")
    assert body == b"hello\r\n..hidden line\r\n..\r\n.\r\n"
    assert protocol.parse_text(body) == "hello\n.hidden line\n."


def test_parse_request_splits_query_and_ignores_gopher_plus():
    assert protocol.parse_request(b"/~veronica\twho made archie\r\n") == ("/~veronica", "who made archie")
    assert protocol.parse_request(b"/history\t+\r\n") == ("/history", "")
    assert protocol.parse_request(b"\r\n") == ("", "")


def test_parse_menu_round_trip():
    raw = protocol.end_menu([protocol.info("hi"), protocol.item("1", "dir", "/d", "h", 7070)])
    entries = protocol.parse_menu(raw)
    assert [e["type"] for e in entries] == ["i", "1"]
    assert entries[1] == {"type": "1", "display": "dir", "selector": "/d", "host": "h", "port": 7070}


# ---- the hole -----------------------------------------------------------

@pytest.mark.parametrize("selector", ["/../../etc/passwd", "/.gophergeist/summaries.json", "..", "/history/../../x", "/nope.txt"])
def test_resolve_refuses_escapes_hidden_and_missing(hole, selector):
    (hole / ".gophergeist").mkdir()
    (hole / ".gophergeist" / "summaries.json").write_text("{}")
    assert Hole(hole).resolve(selector) is None


def test_resolve_refuses_symlink_out_of_hole(hole, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve")
    (hole / "link.txt").symlink_to(secret)
    assert Hole(hole).resolve("/link.txt") is None


def test_documents_finds_nested_text_and_skips_hidden(hole):
    (hole / ".secret.txt").write_text("hidden")
    selectors = {d.selector for d in Hole(hole).documents()}
    assert "/ofrenda/finger.txt" in selectors and "/welcome.txt" in selectors
    assert "/.secret.txt" not in selectors


# ---- summaries ----------------------------------------------------------

def test_summaries_are_cached_by_content(hole):
    spirit = FakeSpirit("An altar for forgotten protocols")
    s = Summaries(Hole(hole))
    n_docs = len(list(Hole(hole).documents()))
    assert s.build(spirit) == n_docs
    assert len(spirit.calls) == n_docs

    again = Summaries(Hole(hole))  # a fresh process reads the cache from disk
    assert again.build(spirit) == 0
    assert again.get(hole / "welcome.txt") == "An altar for forgotten protocols"

    (hole / "welcome.txt").write_text("Changed!\nNew words.")
    assert again.build(spirit) == 1


def test_summary_falls_back_to_first_line_without_model(hole):
    assert Summaries(Hole(hole)).get(hole / "ofrenda" / "wais.txt") == "WAIS (around 1990)"


# ---- veronica -----------------------------------------------------------

def test_stem_is_crude_but_useful():
    assert stem("licensing") == stem("licensed") == "licens"
    assert stem("gopher") == "gopher"


def test_chunk_text_respects_size():
    text = "\n\n".join(["word " * 100] * 10)
    chunks = chunk_text(text, size=1100)
    assert len(chunks) == 5 and all(len(c) <= 1100 for c in chunks)


def test_index_finds_the_right_document(hole):
    index = Index(Hole(hole).documents())
    assert index.search("Jughead University of Utah")[0].selector == "/history/veronica-and-jughead.txt"
    assert index.search("who made archie")[0].selector == "/ofrenda/archie.txt"
    assert index.search("pizza") == []


def test_ask_sends_numbered_sources_to_the_spirit(hole):
    spirit = FakeSpirit()
    answer = ask("license fees", Index(Hole(hole).documents()), spirit)
    assert answer.spoke and answer.text == spirit.reply
    system, prompt = spirit.calls[0]
    assert "ONLY from the numbered sources" in system
    assert "[1] the-license-that-killed-it.txt" in prompt
    assert prompt.endswith("Question: license fees")
    assert answer.sources[0] == ("/history/the-license-that-killed-it.txt", "the-license-that-killed-it.txt")


def test_ask_falls_back_to_index_when_spirit_sleeps(hole):
    answer = ask("license fees", Index(Hole(hole).documents()), SleepingSpirit())
    assert not answer.spoke
    assert "would not wake" in answer.text and "[1]" in answer.text


def test_ask_with_no_matches_does_not_call_spirit(hole):
    spirit = FakeSpirit()
    answer = ask("pizza", Index(Hole(hole).documents()), spirit)
    assert answer.sources == [] and spirit.calls == []


# ---- server -------------------------------------------------------------

def test_root_menu_has_search_dirs_and_blurbs(hole):
    srv = GopherServer(hole, hostname="ghost.local", port=7070)
    entries = protocol.parse_menu(srv.respond(""))
    links = [(e["type"], e["selector"]) for e in entries if e["type"] != "i"]
    assert ("7", SEARCH) in links and ("1", "/history") in links and ("0", "/welcome.txt") in links
    assert all(e["host"] == "ghost.local" and e["port"] == 7070 for e in entries if e["type"] != "i")
    assert any(e["display"].strip() == "Welcome, traveler" for e in entries)


def test_search_menu_links_to_sources(hole):
    srv = GopherServer(hole, backend=FakeSpirit())
    entries = protocol.parse_menu(srv.respond(SEARCH, "license fees"))
    text = " ".join(e["display"] for e in entries if e["type"] == "i")
    assert "1993 license fees [1]" in text
    assert ("0", "/history/the-license-that-killed-it.txt") in [(e["type"], e["selector"]) for e in entries]


def test_search_without_query_is_an_error(hole):
    assert protocol.parse_menu(GopherServer(hole).respond(SEARCH, ""))[0]["type"] == "3"


def test_index_rebuilds_when_files_change(hole):
    srv = GopherServer(hole)
    assert srv.index().search("cempasuchil") == []
    (hole / "new.txt").write_text("The cempasuchil marigold guides spirits home.")
    assert srv.index().search("cempasuchil")[0].selector == "/new.txt"


def test_missing_selector_is_a_ghostly_error(hole):
    entry = protocol.parse_menu(GopherServer(hole).respond("/geocities/index.txt"))[0]
    assert entry["type"] == "3" and "Only the wind" in entry["display"]


def test_binary_files_are_sent_raw(hole):
    (hole / "pumpkin.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    srv = GopherServer(hole)
    assert ("I", "/pumpkin.png") in [(e["type"], e["selector"]) for e in protocol.parse_menu(srv.respond("/"))]
    assert srv.respond("/pumpkin.png") == b"\x89PNG\r\n\x1a\nfake"


def test_tcp_round_trip_with_the_bundled_client(hole):
    srv = GopherServer(hole, backend=FakeSpirit(), port=0)
    loop = asyncio.new_event_loop()
    server = loop.run_until_complete(srv.start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        menu, links = client.render_menu(protocol.parse_menu(client.fetch("127.0.0.1", srv.port, "/")))
        assert "ASK Ask Veronica" in menu
        doc = protocol.parse_text(client.fetch("127.0.0.1", srv.port, "/ofrenda/finger.txt"))
        assert doc.startswith("finger (1971)")
        answer = protocol.parse_menu(client.fetch("127.0.0.1", srv.port, SEARCH, "license fees"))
        assert any("The wind remembers" in e["display"] for e in answer)
        # A raw path-traversal request over the wire gets an error, not /etc/passwd.
        with socket.create_connection(("127.0.0.1", srv.port)) as s:
            s.sendall(b"/../../../../etc/passwd\r\n")
            assert s.recv(4096).startswith(b"3Nothing lives")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()


def test_browse_follows_links_and_asks(hole, monkeypatch):
    pages = {
        "/": protocol.end_menu([protocol.item("7", "Ask", SEARCH, "h", 70), protocol.item("0", "doc", "/d.txt", "h", 70)]),
        SEARCH: protocol.end_menu([protocol.info("the archives answer")]),
    }
    requested = []

    def fake_fetch(host, port, selector, query=""):
        requested.append((selector, query))
        return pages[selector or "/"]

    monkeypatch.setattr(client, "fetch", fake_fetch)
    answers = iter(["1", "who made archie", "q"])
    printed = []
    client.browse("h", 70, "1", "/", input_fn=lambda _: next(answers), print_fn=printed.append)
    assert requested == [("/", ""), (SEARCH, "who made archie")]
    assert "the archives answer" in printed[-1]


def test_parse_url():
    assert client.parse_url("gopher://localhost:7070/0/ofrenda/finger.txt") == ("localhost", 7070, "0", "/ofrenda/finger.txt")
    assert client.parse_url("gopher.floodgap.com") == ("gopher.floodgap.com", 70, "1", "")


# ---- model backends against a stub server --------------------------------

class _Stub(BaseHTTPRequestHandler):
    received = []

    def log_message(self, *args):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "llama3.2:3b"}]})
        else:
            self._send({"data": [{"id": "local"}]})

    def _send_lines(self, lines):
        data = "".join(line + "\n" for line in lines).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.received.append((self.path, body))
        if self.path == "/api/generate":
            self._send({"done": True})
        elif self.path == "/api/chat" and body.get("stream"):
            self._send_lines([json.dumps({"message": {"content": w}, "done": False}) for w in ("boo ", "from ", "ollama")]
                             + [json.dumps({"message": {"content": ""}, "done": True})])
        elif self.path == "/api/chat":
            self._send({"message": {"role": "assistant", "content": " boo from ollama "}})
        elif body.get("stream"):
            self._send_lines([": keep-alive"]
                             + ["data: " + json.dumps({"choices": [{"delta": {"content": w}}]}) for w in ("boo ", "from ", "llama.cpp")]
                             + ["data: [DONE]"])
        else:
            self._send({"choices": [{"message": {"content": "boo from llama.cpp"}}]})


@pytest.fixture
def stub():
    _Stub.received = []
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_ollama_backend(stub):
    o = Ollama("llama3.2:3b", stub)
    o.check()
    assert o.generate("sys", "hi", max_tokens=12) == "boo from ollama"
    path, body = _Stub.received[0]
    assert path == "/api/chat" and body["stream"] is False and body["options"]["num_predict"] == 12
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    with pytest.raises(ModelError, match="ollama pull qwen2.5:3b"):
        Ollama("qwen2.5:3b", stub).check()


def test_openai_compatible_backend(stub):
    o = OpenAICompatible("local", stub + "/v1")
    o.check()
    assert o.generate("sys", "hi") == "boo from llama.cpp"
    assert _Stub.received[0][0] == "/v1/chat/completions"


def test_unreachable_model_raises_model_error():
    with pytest.raises(ModelError):
        Ollama("x", "http://127.0.0.1:9").generate("s", "p")


def test_make_backend():
    assert make_backend("none") is None
    assert isinstance(make_backend("openai"), OpenAICompatible)
    with pytest.raises(ValueError):
        make_backend("cloud")


# ---- web view -------------------------------------------------------------

@pytest.fixture
def web(hole):
    from gophergeist.web import WebGateway

    gw = WebGateway(GopherServer(hole, backend=FakeSpirit(), port=7070), port=0)
    gw.serve_in_background()
    yield f"http://127.0.0.1:{gw.port}"
    gw.shutdown()


def _get(url):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers["Content-Type"], r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers["Content-Type"], e.read().decode("utf-8", errors="replace")


def test_web_root_is_self_contained_html(web):
    status, ctype, body = _get(web + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert "<form action='/7/~veronica'>" in body and "href='/1/history'" in body
    # Nothing that would need the internet: no external scripts, styles or fonts.
    assert "http://" not in body.replace("http://127.0.0.1", "") and "https://" not in body
    assert "<script" not in body


def test_web_text_document_is_escaped(web, hole):
    (hole / "trick.txt").write_text("<script>boo()</script>\n.dot line")
    status, _, body = _get(web + "/0/trick.txt")
    assert status == 200
    assert "&lt;script&gt;boo()&lt;/script&gt;" in body and "<script>boo" not in body
    assert ".dot line" in body


def test_web_search_shows_answer_and_source_links(web):
    status, _, body = _get(web + "/7/~veronica?q=license+fees")
    assert status == 200
    assert "The wind remembers" in body
    assert "href='/0/history/the-license-that-killed-it.txt'" in body


def test_web_refuses_traversal(web):
    status, _, body = _get(web + "/0/..%2F..%2Fetc%2Fpasswd")
    assert status == 404 and "Only the wind" in body and "root:" not in body


def test_web_serves_binaries_with_a_content_type(web, hole):
    (hole / "pumpkin.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    status, ctype, _ = _get(web + "/I/pumpkin.png")
    assert status == 200 and ctype == "image/png"


# ---- speed: streaming, warm-up, caching, short prompts --------------------

class StreamingSpirit(FakeSpirit):
    def stream(self, system, prompt, max_tokens=300):
        self.calls.append((system, prompt))
        for word in self.reply.split(" "):
            yield word + " "


class FadingSpirit(FakeSpirit):
    def stream(self, system, prompt, max_tokens=300):
        yield "Gopher was"
        raise ModelError("out of memory")


def test_ollama_streams_and_keeps_model_loaded(stub):
    o = Ollama("llama3.2:3b", stub)
    assert "".join(o.stream("sys", "hi", max_tokens=20)) == "boo from ollama"
    path, body = _Stub.received[-1]
    assert body["stream"] is True and body["keep_alive"] == "30m"
    o.warm()
    assert _Stub.received[-1] == ("/api/generate", {"model": "llama3.2:3b", "keep_alive": "30m"})


def test_openai_compatible_streams_server_sent_events(stub):
    o = OpenAICompatible("local", stub + "/v1")
    assert list(o.stream("sys", "hi")) == ["boo ", "from ", "llama.cpp"]
    assert _Stub.received[-1][1]["stream"] is True


def test_prompt_is_short_and_focused(hole):
    from gophergeist.veronica import PASSAGE_CHARS, TOP_K, prepare

    chunks, sources, prompt = prepare("who wrote jughead", Index(Hole(hole).documents()))
    assert len(chunks) <= TOP_K
    assert "Rhett" in prompt
    # Three passages of at most ~450 characters each, plus a little framing.
    assert len(prompt) < TOP_K * (PASSAGE_CHARS + 120)


def test_ask_stream_yields_pieces(hole):
    from gophergeist.veronica import ask_stream

    sources, pieces = ask_stream("license fees", Index(Hole(hole).documents()), StreamingSpirit())
    parts = list(pieces)
    assert len(parts) > 3 and "".join(parts).strip() == StreamingSpirit().reply
    assert sources[0][0] == "/history/the-license-that-killed-it.txt"


def test_ask_stream_survives_a_spirit_that_fades(hole):
    from gophergeist.veronica import ask_stream

    _, pieces = ask_stream("license fees", Index(Hole(hole).documents()), FadingSpirit())
    assert "".join(pieces) == "Gopher was ... (the spirit faded mid-sentence)"


def test_repeat_questions_are_answered_from_cache(hole):
    spirit = FakeSpirit()
    srv = GopherServer(hole, backend=spirit)
    srv.respond(SEARCH, "license fees")
    srv.respond(SEARCH, "  License   FEES ")
    assert len(spirit.calls) == 1
    (hole / "welcome.txt").write_text("edited, so cached answers are stale")
    srv.respond(SEARCH, "license fees")
    assert len(spirit.calls) == 2


def test_fallback_answers_are_not_cached(hole):
    srv = GopherServer(hole, backend=SleepingSpirit())
    srv.respond(SEARCH, "license fees")
    assert srv.cached_answer("license fees") is None


def test_streamed_answers_are_cached_too(hole):
    spirit = StreamingSpirit()
    srv = GopherServer(hole, backend=spirit)
    "".join(srv.answer_stream("license fees")[1])
    _, pieces = srv.answer_stream("license fees")
    assert "".join(pieces).strip() == spirit.reply and len(spirit.calls) == 1


def test_blurbs_wait_while_a_visitor_asks(hole):
    srv = GopherServer(hole, backend=FakeSpirit())
    order = []
    with srv.question():
        t = threading.Thread(target=lambda: (srv.wait_until_idle(), order.append("blurb")))
        t.start()
        t.join(0.2)
        order.append("answer")
    t.join(2)
    assert order == ["answer", "blurb"]


def test_web_streams_the_answer(hole):
    import urllib.request
    from gophergeist.web import WebGateway

    gw = WebGateway(GopherServer(hole, backend=StreamingSpirit(), port=7070), port=0)
    gw.serve_in_background()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{gw.port}/7/~veronica?q=license+fees", timeout=5) as r:
            assert r.headers["Content-Length"] is None  # streamed, not buffered
            body = r.read().decode()
    finally:
        gw.shutdown()
    assert "the spirit stirs" in body and "1993 license fees [1]." in body
    assert "href='/0/history/the-license-that-killed-it.txt'" in body and body.endswith("</html>")


# ---- fine-tuning dataset ---------------------------------------------------

class Teacher:
    """Plays the big teacher model: writes questions, then answers like a well-behaved Veronica."""

    def __init__(self):
        self.calls = []

    def generate(self, system, prompt, max_tokens=300):
        from gophergeist.dataset import QUESTION_SYSTEM

        self.calls.append(system)
        if system == QUESTION_SYSTEM:
            topic = prompt.split("Passage from ", 1)[1].split(".", 1)[0].replace("-", " ")
            return f"1. Who created {topic}?\n- When did {topic} happen?\nWhat was {topic} for?\nok"
        if "pizza" in prompt.lower() or "Mona Lisa" in prompt:
            return "The archives are silent on that."
        return "It began in the early nineties at a university [1]. The dust remembers."


def test_dataset_examples_use_the_runtime_prompt(hole, tmp_path):
    from gophergeist.dataset import build, write_jsonl
    from gophergeist.veronica import SYSTEM

    docs = list(Hole(hole).documents())
    examples = build(docs, Teacher(), per_passage=3, mismatched=0)
    assert len(examples) >= len(docs) * 3 * 0.8
    for ex in examples:
        system, user, assistant = ex["messages"]
        assert system == {"role": "system", "content": SYSTEM}
        assert user["content"].startswith("Sources:\n[1] ") and "\n\nQuestion: " in user["content"]
        assert assistant["role"] == "assistant"
    n_train, n_eval = write_jsonl(examples, tmp_path / "t.jsonl", tmp_path / "e.jsonl")
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == n_train and n_eval == len(examples) // 10
    assert json.loads(lines[0])["messages"][2]["content"]


def test_dataset_teaches_refusals_and_drops_bad_answers():
    from gophergeist.dataset import good_answer, parse_questions

    assert good_answer("Gopher began in 1991 [1].", answerable=True)
    assert not good_answer("Gopher began in 1991.", answerable=True)  # no citation
    assert good_answer("The archives are silent on that.", answerable=False)
    assert not good_answer("The archives are silent, but [1] says pizza.", answerable=False)
    assert not good_answer("## Answer\nGopher [1]", answerable=True)  # markdown
    assert parse_questions("1. Who made Archie?\n- why?\n* What is WAIS exactly?", 5) == ["Who made Archie?", "What is WAIS exactly?"]


# ---- séance mode (internet sources are faked; nothing here touches the network) ----

def fake_spirits(query):
    from gophergeist.seance import Spirit

    fake_spirits.asked.append(query)
    return [
        Spirit("Paris", "Paris is the capital and largest city of France.", "URL:https://en.wikipedia.org/wiki/Paris",
               "Wikipedia"),
        Spirit("Travel notes: France", "The capital of France is Paris, on the Seine.",
               "gopher://gopher.example.org:70/0/phlog/france.txt", "Gopherspace"),
    ]


fake_spirits.asked = []


def test_needs_seance_only_when_the_hole_cannot_answer(hole):
    from gophergeist.veronica import needs_seance

    index = Index(Hole(hole).documents())
    assert needs_seance("what is the capital of France", index.search("what is the capital of France"))
    assert not needs_seance("who wrote jughead", index.search("who wrote jughead"))
    assert not needs_seance("the", [])  # nothing left after stopwords


def test_seance_reaches_beyond_when_archives_are_silent(hole):
    from gophergeist.veronica import SEANCE_SYSTEM

    fake_spirits.asked = []
    spirit = FakeSpirit("Beyond the veil, a voice: Paris is the capital of France [1][2].")
    answer = ask("what is the capital of France", Index(Hole(hole).documents()), spirit, fake_spirits)
    system, prompt = spirit.calls[0]
    assert system == SEANCE_SYSTEM and prompt.startswith("Sources summoned from beyond the veil:")
    assert "[1] Paris (Wikipedia): Paris is the capital" in prompt
    assert answer.spoke and answer.sources[0] == ("URL:https://en.wikipedia.org/wiki/Paris", "Paris (Wikipedia)")


def test_seance_stays_home_when_the_files_know(hole):
    fake_spirits.asked = []
    ask("who wrote jughead", Index(Hole(hole).documents()), FakeSpirit(), fake_spirits)
    assert fake_spirits.asked == []


def test_seance_is_off_by_default(hole):
    srv = GopherServer(hole, backend=FakeSpirit())
    assert srv.summon is None
    assert "silent" in protocol.parse_menu(srv.respond(SEARCH, "what is the capital of France"))[2]["display"]


def test_seance_without_internet_falls_back_to_silence(hole):
    from gophergeist.seance import summon

    def offline(query):
        raise OSError("no route to host")

    assert summon("anything", sources=(offline,)) == []
    answer = ask("capital of France", Index(Hole(hole).documents()), FakeSpirit(), lambda q: summon(q, (offline,)))
    assert "silent" in answer.text


def test_seance_menu_links_to_the_web_and_gopherspace(hole):
    srv = GopherServer(hole, backend=FakeSpirit("Paris [1]."), hostname="ghost.local", port=7070, seance=fake_spirits)
    entries = protocol.parse_menu(srv.respond(SEARCH, "what is the capital of France"))
    links = {(e["type"], e["selector"], e["host"]) for e in entries if e["type"] not in "i7"}
    assert ("h", "URL:https://en.wikipedia.org/wiki/Paris", "ghost.local") in links
    assert ("0", "/phlog/france.txt", "gopher.example.org") in links
    assert b"https://en.wikipedia.org/wiki/Paris" in srv.respond("URL:https://en.wikipedia.org/wiki/Paris")


def test_seance_web_view_links_out(hole):
    from gophergeist.web import WebGateway

    gw = WebGateway(GopherServer(hole, backend=StreamingSpirit("Paris [1]."), port=7070, seance=fake_spirits), port=0)
    gw.serve_in_background()
    try:
        status, _, body = _get(f"http://127.0.0.1:{gw.port}/7/~veronica?q=capital+of+France")
    finally:
        gw.shutdown()
    assert status == 200
    assert "href='https://en.wikipedia.org/wiki/Paris'" in body
    assert "href='gopher://gopher.example.org:70/0/phlog/france.txt'" in body


def test_search_gopherspace_reads_veronica2_results():
    from gophergeist.seance import VERONICA2, search_gopherspace

    menu = protocol.end_menu([
        protocol.info("Veronica-2 results"),
        protocol.item("0", "Gopher history", "/history.txt", "gopher.example.org", 70),
        protocol.item("1", "A haunted phlog", "/phlog", "ghost.example.org", 7070),
    ])
    calls = []

    def fetch(host, port, selector, query="", timeout=0):
        calls.append((host, selector, query))
        return menu if selector == VERONICA2[2] else protocol.text_body("Gopher was born in Minnesota.")

    spirits = search_gopherspace("gopher history", fetch=fetch)
    assert calls[0] == ("gopher.floodgap.com", "/v2/vs", "gopher history")
    assert spirits[0].text == "Gopher was born in Minnesota." and spirits[0].origin == "Gopherspace"
    assert spirits[1].selector == "gopher://ghost.example.org:7070/1/phlog"


def test_search_wikipedia_uses_summaries():
    from gophergeist.seance import search_wikipedia

    def get_json(url):
        if "list=search" in url:
            assert "srsearch=capital+of+France" in url
            return {"query": {"search": [{"title": "Paris"}, {"title": "Empty Page"}]}}
        if url.endswith("Paris"):
            return {"extract": "Paris is the capital of France.",
                    "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Paris"}}}
        return {}

    spirits = search_wikipedia("capital of France", get_json=get_json)
    assert [(s.title, s.selector) for s in spirits] == [("Paris", "URL:https://en.wikipedia.org/wiki/Paris")]
