"""A tiny Gopher client, so you can visit the hole without installing Lynx."""

import socket
from urllib.parse import unquote, urlsplit

from . import protocol

LINK_TYPES = {"0": "TXT", "1": "DIR", "7": "ASK", "9": "BIN", "g": "GIF", "I": "IMG", "h": "WEB"}


def parse_url(url):
    """gopher://host[:port]/[type][selector] -> (host, port, type, selector)."""
    if "://" not in url:
        url = "gopher://" + url
    parts = urlsplit(url)
    path = unquote(parts.path)
    itype, selector = "1", ""
    if len(path) > 1:
        itype, selector = path[1], path[2:]
    return parts.hostname or "localhost", parts.port or 70, itype, selector


def fetch(host, port, selector, query="", timeout=120):
    request = selector + (f"\t{query}" if query else "") + "\r\n"
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(request.encode("utf-8"))
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks)


def render_menu(entries):
    """Menu as terminal text, plus the list of followable links (numbered from 1)."""
    out, links = [], []
    for e in entries:
        if e["type"] == "i":
            out.append(f"       {e['display']}")
        elif e["type"] == "3":
            out.append(f"  !!   {e['display']}")
        else:
            links.append(e)
            tag = LINK_TYPES.get(e["type"], "???")
            out.append(f"{len(links):>3}. {tag} {e['display']}")
    return "\n".join(out), links


def browse(host, port, itype="1", selector="", query="", input_fn=input, print_fn=print):
    """A minimal interactive browser: type a number to follow it, b to go back, q to quit."""
    history = []
    page = (host, port, itype, selector, query)
    while True:
        h, p, t, sel, q = page
        raw = fetch(h, p, sel, q)
        links = []
        if t in ("1", "7"):
            text, links = render_menu(protocol.parse_menu(raw))
            print_fn(text)
        elif t == "0":
            print_fn(protocol.parse_text(raw))
        else:
            print_fn(f"({len(raw)} bytes of type {t}; this little client only shows text)")
        choice = input_fn("\n[number] follow  [b]ack  [q]uit > ").strip().lower()
        if choice == "q":
            return
        if choice == "b":
            if history:
                page = history.pop()
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(links):
            e = links[int(choice) - 1]
            q = input_fn("Ask: ").strip() if e["type"] == "7" else ""
            history.append(page)
            page = (e["host"], e["port"], e["type"], e["selector"], q)
