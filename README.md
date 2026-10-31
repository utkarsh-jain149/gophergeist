# 👻 gophergeist

**A Gopher server haunted by an open-weight model. Veronica, Gopherspace's long-dead search engine, answers again.**

Built for the DEV **Ghost in the Machine** challenge (Halloween and Día de los Muertos week): build something
spooky, or something that honors the past, with open-source AI at its core.

Before the web won, a lot of the internet looked like this: plain menus, plain text, nothing that moved. It was
called **Gopher**, it came out of the University of Minnesota in 1991, and for a couple of years it was the most
popular way to find things online. Then it mostly died.

`gophergeist` brings it back. Point it at any folder of text files and it becomes a real, standards-compliant
gopherhole that works in `curl`, Lynx, or any Gopher client from the last 30 years. And it raises a ghost:
**Veronica**, Gopherspace's 1992 search engine, now answers questions about the hole using a **small open-weight
model running on your own machine**, and cites the documents she read.

```
$ gophergeist serve demo/hole
gophergeist 0.1.0 haunting /home/me/gophergeist/demo/hole
  spirit: llama3.2:3b via Ollama
  writing menu blurbs for 10 documents in the background...
  listening on gopher://localhost:7070/
  try: curl gopher://localhost:7070/   or   lynx gopher://localhost:7070/

$ gophergeist browse
          .-.
         (o o)   g o p h e r g e i s t
         | O \   a gopherhole, haunted by an open-weight model
          \   \
           `~~~'

  1. ASK Ask Veronica (search, answered by a local ghost)

  2. DIR history/
  3. DIR ofrenda/
  4. TXT welcome.txt
           Welcome, traveler

  5. TXT About this ghost

[number] follow  [b]ack  [q]uit >
```

Ask her something:

```
$ gophergeist browse --ask "who wrote jughead?"
       Veronica, asked: "who wrote jughead?"

       Jughead was written in 1993 by Rhett "Jonzy" Jones at the
       University of Utah [1]. Unlike me, he only searched a few servers
       at a time [1]. I remember him well. Nobody asks for him anymore.

       Sources:
  1. TXT [1] veronica-and-jughead.txt
  2. TXT [2] finger.txt
  3. TXT [3] archie.txt

  4. ASK Ask another question
  5. DIR Back to the main menu
```

> The answer above is illustrative: it shows the shape of a model's reply, and the wording depends on your model.
> The other transcripts in this README were captured from real runs.

With the model switched off (`--backend none`), Veronica still answers, the way the 1992 original did: straight
from the index.

```
       No spirit is bound to this server, so the index speaks alone:

       [1] Jughead followed in 1993, written by Rhett "Jonzy" Jones at the
       University of Utah.
       [2] Les Earnest wrote the first version around 1971 at the Stanford
       Artificial Intelligence Laboratory.
       [3] Veronica and Jughead, Gopher's search engines, took their names
       from Archie's comic book friends.
```

## What's in the hole

The bundled `demo/hole` is a small gopherhole about Gopher itself:

- **`history/`**: how Gopher was born, the item types, Veronica and Jughead, and the licensing decision that sank it.
- **`ofrenda/`**: for Día de los Muertos, an altar for protocols the internet left behind (Archie, WAIS, finger),
  each with who made it, what it gave us, and a favorite food to leave on the altar.

Use your own folder instead: notes, a zine, a family history, an old blog export. Anything that's text.

## Quick start

You need Python 3.9+ and a local model server. No other dependencies, no API keys, no accounts.

```bash
# 1. A local open-weight model (any Ollama model works)
ollama pull llama3.2:3b

# 2. gophergeist
git clone <this repo> && cd gophergeist
pip install -e .

# 3. Haunt a folder
gophergeist serve demo/hole

# 4. Visit it (any of these)
gophergeist browse
gophergeist serve demo/hole --web     # then open http://localhost:8070 in a browser
curl gopher://localhost:7070/
lynx gopher://localhost:7070/
curl "gopher://localhost:7070/7/~veronica%09what%20killed%20gopher"
```

### Not using Ollama?

Anything that speaks the OpenAI-compatible chat API works, which covers **llama.cpp's `llama-server`,
llamafile, LM Studio and vLLM**:

```bash
# For example, a llamafile: one executable with the weights inside. Download once, run forever.
./Llama-3.2-3B-Instruct.Q6_K.llamafile --server --nobrowser   # serves on :8080

gophergeist serve demo/hole --backend openai --model-host http://localhost:8080/v1
```

`gophergeist doctor` checks that your model answers. `gophergeist ask demo/hole "question"` asks Veronica
without running a server.

## Works with the internet off

Everything gophergeist needs lives on your machine, so after a one-time download it runs with no connection at all:

1. **While online, once:** `pip install -e .` and `ollama pull llama3.2:3b` (or download a single llamafile).
2. **Unplug.** Turn off Wi-Fi.
3. `gophergeist serve demo/hole --web` and open `http://localhost:8070`, or use `gophergeist browse`.

Menus, documents, Veronica's answers and the menu blurbs all keep working. The web view is a single HTML page per
request with inline CSS: no CDN, no web fonts, no JavaScript, no analytics. A test checks that its pages reference
nothing outside your machine, and in a headless browser the pages made zero requests to anything but localhost.

For a fully portable kit, put this repo and a llamafile on a USB stick. Plug it into any laptop, start the
llamafile, then run `gophergeist serve FOLDER --backend openai --web`. No install beyond Python, and nothing to
sign in to.

## What hardware do I need?

No GPU required. A GPU only makes answers faster.

| Your machine | Suggested setup |
| --- | --- |
| Laptop or desktop with an NVIDIA/Apple GPU | `llama3.2:3b` (the default), or bigger |
| Ordinary laptop, CPU only, 8 GB RAM | `--model llama3.2:1b` or `--model gemma3:1b` |
| Old laptop, Raspberry Pi 5, a $5 VPS | a 1B model, or `--backend none` (index only, instant) |

### Making CPU answers feel fast

On a CPU, how long an answer takes depends mostly on how much text the model has to read and how much it
writes. gophergeist keeps both small and hides the rest:

- **Short prompts.** Veronica reads only the three best passages, each trimmed to the ~450 characters most related to
  the question, behind a one-line system prompt.
- **Short answers.** About 80 words, capped at 160 tokens.
- **Streaming.** In the web view, words appear as the model writes them, so you see the start of the answer in a
  second or two instead of staring at a blank page.
- **Warm model.** The model is loaded when the server starts and kept in memory for 30 minutes between questions,
  so nobody waits for weights to load from disk.
- **Cached answers.** Repeat questions come back instantly. The cache resets itself when you edit a file.
- **Visitors first.** Menu blurbs are written in the background and pause whenever someone asks a question.

These are design choices, not benchmarks. Your numbers will depend on your CPU and model, so time a few questions
with `gophergeist ask` before you promise anyone a speed.

### Fine-tune a faster Veronica

A stock 1B model is fast but sometimes forgets to cite or wanders off the sources. Fine-tuning fixes that: a
bigger open teacher model writes a few hundred example answers in Veronica's style from your own gopherhole, and
a 1B student learns the behaviour. It learns how to answer, not what the facts are, so editing your files never
needs retraining.

```bash
gophergeist dataset demo/hole --model qwen2.5:7b    # writes veronica-train.jsonl / veronica-eval.jsonl
```

`finetune/veronica_colab.ipynb` runs the whole pipeline on a free Colab GPU: teacher, LoRA training with Unsloth,
and export to a GGUF you load into Ollama as `llama3.2-veronica:1b`. See [`finetune/`](finetune/README.md).

## Commands

| Command | What it does |
| --- | --- |
| `gophergeist serve FOLDER` | Serve a folder as a gopherhole on port 7070. Writes menu blurbs in the background. `--web` adds a browser view on port 8070. `--bind 0.0.0.0` shares it on your network; `--port 70` is traditional (needs root). |
| `gophergeist browse [URL]` | A tiny interactive Gopher client. Works on any gopherhole, not just this one. `--ask "..."` goes straight to Veronica. |
| `gophergeist ask FOLDER "question"` | Ask Veronica from the command line. |
| `gophergeist index FOLDER` | Write all menu blurbs now instead of in the background. |
| `gophergeist dataset FOLDER...` | Write fine-tuning data that teaches a small model to be Veronica (see `finetune/`). |
| `gophergeist doctor` | Check that the local model is reachable and answers. |

Every model command takes `--backend ollama|openai|none`, `--model` and `--model-host`.

## How the open pieces make it work

| Piece | What it does |
| --- | --- |
| **Gopher, RFC 1436** (open standard, 1993) | The whole wire protocol fits on a page: one request line in, a menu or a document out. The server is under 200 lines of `asyncio`. |
| **Ollama** or **llama.cpp / llamafile** (open source) | Runs the model on `localhost`. gophergeist talks to them over HTTP with `urllib`, no SDK. |
| **Llama 3.2 3B** (open weights, default) | Gives Veronica her voice, and writes the one-line blurb under each menu entry. Swap in `qwen2.5:3b`, `gemma3`, `phi4-mini`, or anything else your machine can run. |
| **BM25** (plain Python) | Finds the passages first. The model only reads the top few and must cite them as `[1]`, `[2]`, so a 3B model stays grounded instead of making things up. |

### The index finds, the ghost speaks

Small local models are good at reading a few paragraphs and answering a question about them. They are not good at
remembering facts about 1990s internet history. So the split is strict:

- A BM25 index (with a deliberately crude stemmer) picks the four best passages from the hole.
- Veronica's system prompt tells the model to answer **only** from those numbered sources, cite them, and say "the
  archives are silent" when they don't cover the question. She gets one ghostly aside, no more.
- Every answer ends with real Gopher links to the sources, so you can check her.
- If the model is asleep, slow, or not installed, the index answers on its own. A ghost should never take the
  server down with it.

### It's a real Gopher server

- Correct type `0`, `1`, `3`, `7`, `9`, `g` and `I` items, plus the near-universal `i` info lines.
- Text documents are dot-stuffed and terminated with a lone `.` as RFC 1436 asks.
- Gopher+ clients that send `\t+` get plain Gopher back.
- Selectors can't escape the folder: `..`, hidden files and symlinks pointing outside are refused (and tested).
- The search index rebuilds itself when you add or edit a file. Blurbs are cached by content hash in
  `.gophergeist/`, which is never served.

## Why open matters for this

**Gopher is the cautionary tale.** In 1993 the University of Minnesota announced license fees for some uses of its
Gopher server. That same spring, on April 30, CERN put the World Wide Web into the public domain. Developers and
companies picked the one nobody could charge them for, and Gopher withered. The web won because it was open.

So it would be a strange tribute to resurrect Gopher on top of an API that someone else can price, change or shut
down. A revival that depends on a closed service is just a ghost waiting to happen. Here is what the open pieces
make possible:

- **It runs with the lights (and the internet) off.** The model, the index and the server all live on one laptop.
  Unplug the router, start `gophergeist serve`, and Veronica still answers. Fitting for a protocol that
  was built for slow campus networks.
- **It can't be taken away.** A llamafile is a single executable with the weights inside. Put it on a USB stick
  next to this repo and the whole thing will still run in 2046, with no account to expire and no endpoint to be
  deprecated. Preservation is the point of a project like this.
- **Your folder stays yours.** People will point this at personal things: journals, family stories, notes. None of
  it is uploaded anywhere to be summarized.
- **Swap the ghost, change the haunting.** Want a bigger model, a multilingual one for a family archive in Spanish,
  or a fine-tune that talks like a 1993 sysadmin? Change `--model`. The persona is a plain string in
  `veronica.py`, so you can rewrite who's haunting the machine.
- **You can train your own ghost.** With the weights in hand, a free Colab GPU turns a 1B model into a Veronica
  that knows the job, and you keep the result as a single file. A closed API lets you rent a fine-tune; it doesn't
  give you the model to keep on a USB stick.
- **It costs nothing to leave running.** A gopherhole is meant to sit there for years. Per-token pricing and a
  server that's always on don't mix; a 3B model on hardware you already own does.

### Where open worked better than closed

- **A tiny model is enough.** Retrieval does the finding, so the model only has to read four short passages and
  answer from them, which is well within what a 3B local model handles. The citations make it easy to see when it
  strays. A bigger hosted model would mostly be paying for knowledge Veronica is told not to use.
- **Five runtimes for the price of one.** Because the open ecosystem converged on the same small HTTP APIs, one
  hundred-line `model.py` supports Ollama, llama.cpp, llamafile, LM Studio and vLLM, with no SDK.
- **Honest trade-off:** a frontier hosted model writes more polished prose and knows more history. But for a
  tool whose job is to answer from *your* files, that extra knowledge is a liability, and the price is your data
  leaving your machine.

## Testing

```bash
pip install pytest
python -m pytest -q
```

The 49 tests run fully offline in a few seconds. A fake "spirit" stands in for the model, and a stub HTTP server
stands in for Ollama and llama.cpp so the real request code is exercised. There's also a real TCP round trip
through the bundled client, plus path-traversal attempts over Gopher and over the web view.

**A note on how this was built:** the sandbox I built it in couldn't download model weights (no access to Ollama's
registry or Hugging Face), so I tested it end to end against a stub Ollama server and in index-only mode, not
against a real model. Run it with a real model before you trust the answer transcript above.

## Limitations

- Text only for answers. Images and binaries are served, but Veronica can't read them.
- BM25 matches words, not meaning: "why did Gopher die?" won't match "killed" as well as "license" does. The model
  smooths this over when it's running.
- No `gophermap` support yet; menus are generated from the folder layout.
- No TLS, which is period-accurate but means you shouldn't serve anything private beyond your LAN.

## History sources

[RFC 1436](https://www.rfc-editor.org/rfc/rfc1436) (The Internet Gopher Protocol, 1993), and the
[Floodgap gopher server](gopher://gopher.floodgap.com/), which has kept Gopherspace and Veronica-2 alive for years.
The demo hole is my own summary; corrections welcome.

## License

MIT. Unlike Gopher in 1993, you will never be asked for a license fee.
