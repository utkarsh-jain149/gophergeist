"""Séance mode: when the archives are silent, Veronica reaches beyond the veil.

Off by default. With ``--seance``, a question the local files can't answer is sent to two
open sources that need no account or API key:

* Veronica-2, the search engine that still indexes the living Gopherspace, at Floodgap.
* Wikipedia's public search API.

The local model still writes the answer, in Veronica's voice, citing what was summoned.
Nothing else leaves the machine: only the question text, and only when the files are silent.
"""

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import client, protocol

VERONICA2 = ("gopher.floodgap.com", 70, "/v2/vs")
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php?"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = "gophergeist/0.1 (a local Gopher server; seance mode)"
TIMEOUT = 12


@dataclass
class Spirit:
    title: str
    text: str
    selector: str  # "gopher://host:port/T/selector" or "URL:https://..."
    origin: str  # shown to the model and the visitor: "Gopherspace" or "Wikipedia"


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def search_gopherspace(query, fetch=client.fetch, limit=3, read=1):
    """Ask Veronica-2 and read the first matching text document or two."""
    host, port, selector = VERONICA2
    entries = [e for e in protocol.parse_menu(fetch(host, port, selector, query, timeout=TIMEOUT))
               if e["type"] in ("0", "1")][:limit]
    spirits = []
    for e in entries:
        text = e["display"]
        if e["type"] == "0" and read > 0:
            read -= 1
            try:
                text = protocol.parse_text(fetch(e["host"], e["port"], e["selector"], timeout=TIMEOUT))[:3000]
            except OSError:
                pass
        link = f"gopher://{e['host']}:{e['port']}/{e['type']}{e['selector']}"
        spirits.append(Spirit(e["display"].strip(), text, link, "Gopherspace"))
    return spirits


def search_wikipedia(query, get_json=_get_json, limit=2):
    params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                     "srlimit": limit, "format": "json"})
    hits = get_json(WIKI_SEARCH + params).get("query", {}).get("search", [])
    spirits = []
    for hit in hits[:limit]:
        title = hit["title"]
        summary = get_json(WIKI_SUMMARY + urllib.parse.quote(title.replace(" ", "_"), safe=""))
        url = summary.get("content_urls", {}).get("desktop", {}).get("page") \
            or "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        if summary.get("extract"):
            spirits.append(Spirit(title, summary["extract"], "URL:" + url, "Wikipedia"))
    return spirits


def summon(query, sources=(search_gopherspace, search_wikipedia)):
    """Everything the spirits will say about query. A source that's unreachable is skipped quietly."""
    spirits = []
    for source in sources:
        try:
            spirits += source(query)
        except Exception:
            continue  # no internet, a slow server, an odd response: the séance goes on without it
    return spirits
