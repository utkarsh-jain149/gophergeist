"""A local web view of the gopherhole, for visitors without a Gopher client.

It translates browser requests into Gopher selectors and renders the answers as plain HTML.
Everything is inline: no CDN, no web fonts, no JavaScript, so it works with the internet off.
"""

import html
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

from . import protocol
from .server import SEARCH

TAGS = {"0": "TXT", "1": "DIR", "7": "ASK", "9": "BIN", "g": "GIF", "I": "IMG", "h": "WEB"}

CSS = """
:root { --bg: #0b0d0b; --fg: #b8f5b0; --dim: #6f9a6a; --link: #f5c26b; --err: #ff8a7a; }
@media (prefers-color-scheme: light) {
  :root { --bg: #f6f3e8; --fg: #1f2a1d; --dim: #6b7562; --link: #8a4b00; --err: #a3261b; }
}
html, body { margin: 0; background: var(--bg); color: var(--fg); }
body { font: 15px/1.45 ui-monospace, "SF Mono", Menlo, Consolas, monospace; padding: 24px 16px; }
main { max-width: 74ch; margin: 0 auto; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: inherit; }
a { color: var(--link); text-decoration: none; }
a:hover, a:focus { text-decoration: underline; }
.tag { color: var(--dim); }
.err { color: var(--err); }
form { display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0; }
input { flex: 1 1 24ch; font: inherit; color: var(--fg); background: transparent;
        border: 1px solid var(--dim); padding: 4px 8px; }
button { font: inherit; color: var(--bg); background: var(--link); border: 0; padding: 4px 12px; cursor: pointer; }
footer { margin-top: 32px; color: var(--dim); font-size: 13px; }
"""


BACK = "<pre><a href='/'>&larr; main menu</a>\n\n</pre>"


def href(itype, selector):
    return "/" + itype + quote(selector)


def page_head(title):
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
    )


PAGE_TAIL = ("<footer>A web view of a Gopher hole, served by gophergeist from this machine.</footer>"
             "</main></body></html>")


def page(title, body):
    return (page_head(title) + body + PAGE_TAIL).encode("utf-8")


def render_menu(entries, gopher_host, gopher_port):
    lines = []
    for e in entries:
        t, display = e["type"], html.escape(e["display"])
        if t == "i":
            lines.append(display)
        elif t == "3":
            lines.append(f"<span class=err>{display}</span>")
        elif t == "7":
            lines.append(f"<form action='{href('7', e['selector'])}'><input name=q aria-label='{display}' "
                         f"placeholder='{display}' required><button>Ask</button></form>")
        else:
            if t == "h" and e["selector"].startswith("URL:"):
                url = e["selector"][4:]
            elif (e["host"], e["port"]) == (gopher_host, gopher_port):
                url = href(t, e["selector"])
            else:  # a link to some other gopherhole: hand it to the visitor's own Gopher client
                url = f"gopher://{e['host']}:{e['port']}/{t}{quote(e['selector'])}"
            lines.append(f"<span class=tag>{TAGS.get(t, '???')}</span> <a href='{html.escape(url)}'>{display}</a>")
    return "<pre>" + "\n".join(lines) + "</pre>"


class WebGateway:
    def __init__(self, gopher, bind="127.0.0.1", port=8070):
        self.gopher = gopher
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                stream = gateway.search_stream(self.path)
                if stream is not None:
                    # No Content-Length: the answer is written as the model produces it, and the
                    # closed connection (HTTP/1.0) marks the end. Browsers render it as it arrives.
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    try:
                        for part in stream:
                            self.wfile.write(part)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        stream.close()  # the visitor left; stop the model too
                    return
                status, ctype, body = gateway.handle(self.path)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer((bind, port), Handler)
        self.port = self.httpd.server_port

    def handle(self, path):
        """Map a browser path to (status, content type, body). '/1/history' -> menu '/history'."""
        url = urlsplit(path)
        itype, selector = "1", ""
        if len(url.path) > 1:
            itype, selector = url.path[1], unquote(url.path[2:])
        query = parse_qs(url.query).get("q", [""])[0].strip()
        data = self.gopher.respond(selector, query)
        title = f"gophergeist {selector or '/'}"

        if itype in ("1", "7"):
            entries = protocol.parse_menu(data)
            status = 404 if entries and entries[0]["type"] == "3" else 200
            body = render_menu(entries, self.gopher.hostname, self.gopher.port)
            if selector not in ("", "/"):
                body = BACK + body
            return status, "text/html; charset=utf-8", page(title, body)
        if itype == "0":
            if data.startswith(b"3"):  # the gopher side answered with an error menu
                return 404, "text/html; charset=utf-8", page(title, render_menu(protocol.parse_menu(data), None, None))
            text = html.escape(protocol.parse_text(data))
            return 200, "text/html; charset=utf-8", page(title, BACK + f"<pre>{text}</pre>")
        if data.startswith(b"3Nothing lives"):
            return 404, "text/html; charset=utf-8", page(title, render_menu(protocol.parse_menu(data), None, None))
        ctype = mimetypes.guess_type(selector)[0] or "application/octet-stream"
        return 200, ctype, data

    def search_stream(self, path):
        """For a Veronica question, a generator of HTML bytes that streams the answer; otherwise None."""
        url = urlsplit(path)
        if url.path != "/7" + quote(SEARCH):
            return None
        query = parse_qs(url.query).get("q", [""])[0].strip()
        if not query:
            return None  # handle() renders the "needs a question" error
        sources, pieces = self.gopher.answer_stream(query)
        gopher = self.gopher

        def generate():
            yield (page_head(f"Veronica: {query}") + BACK
                   + f"<pre>Veronica, asked: \"{html.escape(query)}\"\n\n"
                   + "<span class=tag>(the spirit stirs...)</span>\n\n").encode("utf-8")
            try:
                for piece in pieces:
                    yield html.escape(piece).encode("utf-8")
            finally:
                close = getattr(pieces, "close", None)
                if close:
                    close()
            entries = []
            if sources:
                entries += [protocol.info(), protocol.info(), protocol.info("Sources:")]
                entries += gopher.source_items(sources)
            entries += [protocol.info(), protocol.item("7", "Ask another question", SEARCH, gopher.hostname, gopher.port)]
            menu = render_menu(protocol.parse_menu(protocol.end_menu(entries)), gopher.hostname, gopher.port)
            yield ("</pre>" + menu + PAGE_TAIL).encode("utf-8")

        return generate()

    def serve_in_background(self):
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        return thread

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
