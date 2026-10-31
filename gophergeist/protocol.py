"""RFC 1436 wire format: menu lines, text bodies, and the lone dot that ends them."""

CRLF = "\r\n"
MAX_SELECTOR = 1024


def _clean(value):
    # Tabs and newlines are the protocol's delimiters, so they can never appear inside a field.
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def item(itype, display, selector, host, port):
    return f"{itype}{_clean(display)}\t{_clean(selector)}\t{_clean(host)}\t{port}{CRLF}"


def info(text=""):
    """An 'i' line: not in RFC 1436, but every client since the mid-90s shows it as plain text."""
    return item("i", text, "", "error.host", 1)


def error(text):
    return item("3", text, "", "error.host", 1)


def end_menu(lines):
    return ("".join(lines) + "." + CRLF).encode("utf-8")


def text_body(text):
    """A type 0 document: CRLF lines, a leading '.' doubled, and a terminating '.' line."""
    out = []
    for line in text.splitlines():
        if line.startswith("."):
            line = "." + line
        out.append(line + CRLF)
    out.append("." + CRLF)
    return "".join(out).encode("utf-8")


def parse_request(line):
    """Split a client's request line into (selector, query). Gopher+ markers ('+', '$', '!') are ignored."""
    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
    selector, _, rest = text.partition("\t")
    query = rest.split("\t", 1)[0]
    if query in ("+", "$", "!") or query.startswith("+"):
        query = ""
    return selector[:MAX_SELECTOR], query.strip()


def parse_menu(raw):
    """Parse a menu response into dicts: {type, display, selector, host, port}."""
    entries = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line == ".":
            break
        if not line:
            continue
        parts = line[1:].split("\t")
        parts += [""] * (4 - len(parts))
        port = parts[3].strip()
        entries.append({
            "type": line[0],
            "display": parts[0],
            "selector": parts[1],
            "host": parts[2],
            "port": int(port) if port.isdigit() else 70,
        })
    return entries


def parse_text(raw):
    lines = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line == ".":
            break
        lines.append(line[1:] if line.startswith("..") else line)
    return "\n".join(lines)
