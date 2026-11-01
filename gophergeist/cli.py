"""gophergeist command line: serve, index, ask, browse, doctor."""

import argparse
import asyncio
import sys
from pathlib import Path

from . import __version__, client, protocol
from .hole import Hole, Summaries
from .model import FALLBACK_MODEL, ModelError, Ollama, make_backend
from .server import BANNER, SEARCH, GopherServer
from .veronica import Index, ask
from .seance import summon
from .web import WebGateway

SEANCE_HELP = ("when your files can't answer, let Veronica search Gopherspace (Veronica-2 at Floodgap) "
               "and Wikipedia. Sends the question text to those sites; off by default")
SEANCE_NOTICE = ("  seance mode: questions your files can't answer are sent to gopher.floodgap.com (Veronica-2) "
                 "and en.wikipedia.org")


DEMO_HOLE = Path(__file__).resolve().parent / "demo"


def _folder(folder):
    """The folder to haunt; with none given, the haunted demo hole that ships with gophergeist."""
    return folder or DEMO_HOLE


def _pull_progress(status, completed, total):
    if completed and total:
        print(f"\r  {status}: {completed * 100 // total}% of {total / 1e6:.0f} MB   ", end="", file=sys.stderr, flush=True)
    elif status:
        print(f"\r  {status}{' ' * 30}", file=sys.stderr, flush=True)


def _backend(args, quiet=False):
    """The configured model, or None if it isn't reachable (we never refuse to serve over it).
    With Ollama, a missing model is downloaded on first run: the fine-tuned Veronica by default,
    falling back to a stock 1B model if she can't be pulled."""
    backend = make_backend(args.backend, args.model, args.model_host)
    if backend is None:
        return None
    try:
        if isinstance(backend, Ollama):
            fallbacks = () if args.model else (FALLBACK_MODEL,)
            wanted = backend.model
            if not quiet and not backend.has(wanted):
                print(f"  First run: downloading {wanted} through Ollama (one time only)...", file=sys.stderr)
            backend.ensure(fallbacks, None if quiet else _pull_progress)
            if not quiet and backend.model != wanted:
                print(f"  ({wanted} isn't available, so using {backend.model} instead.)", file=sys.stderr)
        else:
            backend.check()
    except ModelError as e:
        if not quiet:
            print(f"  (no spirit: {e}. Veronica will answer from the index alone.)", file=sys.stderr)
        return None
    return backend


def cmd_serve(args):
    backend = _backend(args)
    server = GopherServer(_folder(args.folder), backend, hostname=args.hostname, port=args.port,
                          summaries=not args.no_summaries, seance=args.seance)
    print()
    for line in BANNER:
        print(line)
    print()
    print(f"gophergeist {__version__} haunting {server.hole.root}")
    print(f"  spirit: {backend or 'none (index only)'}")
    if args.seance:
        print(SEANCE_NOTICE)

    def progress(doc, line):
        print(f"  summarized {doc.selector}: {line}" if doc else f"  {line}")

    server.warm_in_background(progress)
    if not args.no_summaries and backend:
        pending = len(server.summaries.missing())
        if pending:
            print(f"  writing menu blurbs for {pending} documents in the background...")
            server.summarize_in_background(progress)

    async def main():
        srv = await server.start(args.bind)
        print(f"  listening on gopher://{args.hostname}:{server.port}/")
        print(f"  try: curl gopher://{args.hostname}:{server.port}/   or   lynx gopher://{args.hostname}:{server.port}/")
        if args.web is not None:
            web = WebGateway(server, args.bind, args.web)
            web.serve_in_background()
            print(f"  web view (no internet needed): http://{args.hostname}:{web.port}/")
        async with srv:
            await srv.serve_forever()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nThe ghost rests.")


def cmd_index(args):
    backend = _backend(args)
    if backend is None:
        print("Menu blurbs need a model; without one, menus show each file's first line.")
        return 1
    summaries = Summaries(Hole(_folder(args.folder)))
    print(f"Summarizing {_folder(args.folder)} with {backend}...")
    n = summaries.build(backend, lambda doc, line: print(f"  {doc.selector}: {line}"))
    print(f"Wrote {n} new blurbs ({len(summaries.cache)} cached).")
    return 0


def cmd_ask(args):
    backend = _backend(args)
    if args.seance:
        print(SEANCE_NOTICE, file=sys.stderr)
    answer = ask(args.question, Index(Hole(_folder(args.folder)).documents()), backend, summon if args.seance else None)
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for n, (selector, title) in enumerate(answer.sources, 1):
            print(f"  [{n}] {selector}")
    return 0


def cmd_browse(args):
    host, port, itype, selector = client.parse_url(args.url)
    query = ""
    if args.ask:
        itype, selector, query = "7", SEARCH, args.ask
    try:
        if args.ask:
            text, _ = client.render_menu(protocol.parse_menu(client.fetch(host, port, selector, query)))
            print(text)
        else:
            client.browse(host, port, itype, selector)
    except OSError as e:
        print(f"Could not reach gopher://{host}:{port}/ ({e})", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


def cmd_dataset(args):
    from .dataset import build, write_jsonl

    if args.backend == "ollama" and not args.model:
        args.model = "qwen2.5:7b"  # a bigger teacher; the 1B student learns from its answers
    teacher = _backend(args)
    if teacher is None:
        print("Building a dataset needs a teacher model.", file=sys.stderr)
        return 1
    docs = [d for folder in (args.folders or [None]) for d in Hole(_folder(folder)).documents()]
    print(f"Writing Veronica lessons from {len(docs)} documents with {teacher} as the teacher...")
    examples = build(docs, teacher, per_passage=args.per_passage, progress=lambda m: print(f"  {m}"))
    n_train, n_eval = write_jsonl(examples, args.output, args.eval)
    print(f"Wrote {n_train} training examples to {args.output}" + (f" and {n_eval} to {args.eval}" if args.eval else ""))
    if n_train < 150:
        print("  Tip: a few hundred examples work better. Add more folders or raise --per-passage.")
    return 0


def cmd_doctor(args):
    backend = make_backend(args.backend, args.model, args.model_host)
    if backend is None:
        print("Backend is 'none': Veronica answers from the BM25 index only.")
        return 0
    try:
        backend.check()
        reply = backend.generate("Reply with one word.", "Say: boo")
    except ModelError as e:
        print(f"No spirit answered: {e}")
        return 1
    print(f"{backend} is awake. It said: {reply!r}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="gophergeist", description="A Gopher server haunted by a local open-weight model.")
    p.add_argument("--version", action="version", version=f"gophergeist {__version__}")
    model_args = argparse.ArgumentParser(add_help=False)
    model_args.add_argument("--backend", choices=["ollama", "openai", "none"], default="ollama",
                            help="ollama (default), openai = any OpenAI-compatible local server (llama.cpp, llamafile, LM Studio), or none")
    model_args.add_argument("--model", help="model name (default llama3.2:3b for Ollama)")
    model_args.add_argument("--model-host", help="model server URL (default http://localhost:11434 or http://localhost:8080/v1)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", parents=[model_args], help="serve a folder as a gopherhole")
    s.add_argument("folder", nargs="?", help="folder of text files (default: the bundled haunted demo)")
    s.add_argument("--port", type=int, default=7070, help="TCP port (70 is traditional but needs root)")
    s.add_argument("--hostname", default="localhost", help="hostname written into menu links")
    s.add_argument("--bind", default="127.0.0.1", help="address to listen on (0.0.0.0 to share on your network)")
    s.add_argument("--seance", action="store_true", help=SEANCE_HELP)
    s.add_argument("--no-summaries", action="store_true", help="don't ask the model for menu blurbs")
    s.add_argument("--web", type=int, nargs="?", const=8070, metavar="PORT",
                   help="also serve a browser view of the hole (default port 8070)")
    s.set_defaults(func=cmd_serve)

    i = sub.add_parser("index", parents=[model_args], help="write model blurbs for every document now")
    i.add_argument("folder", nargs="?", help="default: the bundled demo")
    i.set_defaults(func=cmd_index)

    a = sub.add_parser("ask", parents=[model_args], help="ask Veronica a question without running a server")
    a.add_argument("question")
    a.add_argument("folder", nargs="?", help="default: the bundled demo")
    a.add_argument("--seance", action="store_true", help=SEANCE_HELP)
    a.set_defaults(func=cmd_ask)

    b = sub.add_parser("browse", help="a tiny interactive Gopher client")
    b.add_argument("url", nargs="?", default="gopher://localhost:7070/")
    b.add_argument("--ask", help="skip the menus and ask the server's Veronica directly")
    b.set_defaults(func=cmd_browse)

    ds = sub.add_parser("dataset", parents=[model_args], help="write fine-tuning data that teaches a small model to be Veronica")
    ds.add_argument("folders", nargs="*", help="folders of text to learn from (default: the bundled demo)")
    ds.add_argument("-o", "--output", default="veronica-train.jsonl")
    ds.add_argument("--eval", default="veronica-eval.jsonl", help="held-out examples for checking the result")
    ds.add_argument("--per-passage", type=int, default=8, help="questions to write per passage")
    ds.set_defaults(func=cmd_dataset)

    d = sub.add_parser("doctor", parents=[model_args], help="check that the local model answers")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
