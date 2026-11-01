*This is a submission for the [Hacktoberfest Open-Source AI Challenge: Ghost in the Machine](https://dev.to/challenges)*
<!-- TODO: swap in the exact challenge link from the submission template -->

# 👻 gophergeist

```
   .-.
  (o o)   g o p h e r g e i s t
  | O \   a gopherhole, haunted by an open-weight model
   \   \
    `~~~'
```

## What I Built

Before the web won, the internet had **Gopher**. Plain menus, plain text, nothing that moved. It came out of the
University of Minnesota in 1991, and for a couple of years it was *the* way to find things online. Then in 1993
the university announced license fees for its server. That same spring CERN put the web in the public domain.
Everyone picked the free one, and Gopher quietly died.

So for Halloween and Día de los Muertos I brought it back from the dead.

**gophergeist** turns any folder of text files into a real Gopher server. It works with `curl`, Lynx, any Gopher
client from the last 30 years, a tiny terminal client I built in, or a retro page in your browser.

And it's haunted. **Veronica** was Gopherspace's actual search engine in 1992. She's back, and she answers your
questions using a small open-weight model running on *your* laptop:

- A keyword index finds the three passages in your files that best match your question.
- The model reads only those, answers in Veronica's voice, and cites them like `[1]`, `[2]`.
- If your files don't cover it, she says **"the archives are silent"** instead of making something up.

I fine-tuned her myself: a 1B Llama model trained to *be* Veronica, small enough to run on a laptop with no GPU and
no internet.

The demo gopherhole is about Gopher itself, written in a haunted voice. There's also an `ofrenda/` folder, an altar for
the protocols the internet buried (Archie, WAIS, finger), each with who made it, what it gave us, and a favourite
food to leave on the altar. 🕯️

## Demo

<!-- TODO: add a GIF or short video: `gophergeist serve --web`, then ask Veronica something -->

```
$ gophergeist serve --web

   .-.
  (o o)   g o p h e r g e i s t
  | O \   a gopherhole, haunted by an open-weight model
   \   \
    `~~~'

gophergeist 0.1.0 haunting .../gophergeist/demo
  spirit: utkarsh2023/llama3.2-veronica:1b via Ollama
  listening on gopher://localhost:7070/
  web view (no internet needed): http://localhost:8070/
```

Asking her something:

```
$ gophergeist browse --ask "who wrote jughead?"
       Veronica, asked: "who wrote jughead?"

       <!-- TODO: paste Veronica's real answer here -->

       Sources:
  1. TXT [1] veronica-and-jughead.txt
  2. TXT [2] finger.txt
  3. TXT [3] archie.txt
```

With no model at all (`--backend none`) she still answers, the way the 1992 original did, straight from the index:

```
       No spirit is bound to this server, so the index speaks alone:

       [1] Jughead followed in 1993, written by Rhett "Jonzy" Jones at the
       University of Utah.
```

### Try it yourself

```bash
# 1. Install Ollama (https://ollama.com/download), then:
pipx install git+https://github.com/utkarsh-jain149/gophergeist

# 2. Raise the ghost, then open http://localhost:8070
gophergeist serve --web
```

If you get `gophergeist: command not found`, run `pipx ensurepath` and open a new terminal.

On the first run it downloads my fine-tuned Veronica (`utkarsh2023/llama3.2-veronica:1b`, ~770 MB) through Ollama,
once. If that fails it falls back to the stock `llama3.2:1b`. To haunt your own notes instead:
`gophergeist serve ~/my-notes --web`.

## Code

{% github utkarsh-jain149/gophergeist %}

```
gophergeist/
  server.py     # the Gopher server (RFC 1436), asyncio
  veronica.py   # BM25 search + the ghost's prompt, streaming and fallbacks
  model.py      # stdlib clients for Ollama and any OpenAI-compatible local server
  web.py        # the retro browser view: inline CSS, no JS, no CDN
  seance.py     # optional: reach Veronica-2 and Wikipedia when your files are silent
  dataset.py    # builds the fine-tuning data
  demo/         # the haunted demo gopherhole
finetune/       # Colab notebook + Ollama Modelfile
tests/          # 63 tests, all offline, with fake models and fake internet
```

Pure Python standard library, no dependencies.

## How I Built It

| Piece | What it does |
| --- | --- |
| **Gopher, RFC 1436** | An open standard from 1993. One request line in, a menu or a document out. |
| **Ollama** | Runs the model on `localhost`. gophergeist talks to it over plain HTTP, no SDK. llama.cpp, llamafile and LM Studio work too (`--backend openai`). |
| **Llama 3.2 1B, fine-tuned** | Veronica's voice. Also writes the one-line blurb under each menu entry. |
| **Qwen 2.5 7B** | The "teacher" that wrote the training data. Only used once, during training. |
| **Unsloth + Google Colab** | LoRA fine-tuning on a free T4 GPU, exported to a 770 MB GGUF. |
| **BM25** | Plain-Python keyword search that finds the passages before the model ever sees them. |

### The index finds, the ghost speaks

Small models are good at reading a few paragraphs and answering from them. They're bad at remembering 1990s
internet history. So the split is strict: search finds the facts, the model only reads and cites. That's what lets
a 1B model give grounded answers, and it means editing your files never needs retraining.

### Fine-tuning Veronica

A stock 1B model sometimes forgets to cite or wanders off the sources. So I taught one to do the job:

1. `gophergeist dataset` has a bigger teacher model (Qwen 2.5 7B) write practice questions from the gopherhole,
   then answer them as Veronica: only from the sources, cite everything, keep it short.
2. Some examples deliberately pair a question with the *wrong* passages, or ask about pizza, so she learns to say
   "the archives are silent".
3. Unsloth LoRA-trains Llama 3.2 1B on that in a free Colab notebook, exports a GGUF, and I pushed it to Ollama.

It learns *how* to answer, not *what* the facts are. The whole pipeline is in
[`finetune/veronica_colab.ipynb`](finetune/veronica_colab.ipynb).

<!-- TODO: stock llama3.2:1b vs fine-tuned, same question: time + did it cite? -->

### Making it fast on a normal laptop

No GPU needed. On a CPU, speed is mostly about how much the model reads and writes, so:

- **Short prompts:** three passages, each trimmed to the ~450 most relevant characters.
- **Short answers:** about 80 words.
- **Streaming:** in the browser, words appear as she writes them.
- **Warm model:** loaded at startup and kept in memory between questions.
- **Cache:** repeat questions come back instantly.
- **Visitors first:** background work pauses while someone is asking.

### Séance mode 🔮

By default Veronica only knows your files. With `--seance`, a question your files can't answer gets carried
"beyond the veil" to **Veronica-2**, the search engine that still indexes the living Gopherspace today, and to
Wikipedia. So the ghost of the 1992 Veronica asks her surviving descendant. It's off by default, only the
question text leaves your machine, and offline she just goes back to "the archives are silent".

## Why Does Open Innovation Matter?

**Gopher is literally the cautionary tale.** It didn't lose because it was worse. It lost because someone could
charge for it, while the web was free to run, change and build on. Resurrecting it on top of a closed API that
someone else can price, change or shut down would be a pretty weird tribute. That would just be another ghost
waiting to happen.

Here's what going open made possible:

- **It works with the lights (and the internet) off.** Model, index and server all live on one laptop. Turn off
  Wi-Fi and Veronica still answers.
- **I could fine-tune her and keep her.** With open weights I trained my own Veronica on a free GPU, and the result
  is one 770 MB file I own. A closed API rents you a fine-tune. It doesn't give you the model to put on a USB stick.
- **Your files stay yours.** People will point this at journals, family stories, notes. None of it is uploaded
  anywhere.
- **It can't be taken away.** No account to expire, no endpoint to get deprecated. A gopherhole is meant to sit
  there for years, and so is this.
- **It costs nothing to leave running.** Per-token pricing and an always-on server don't mix. A 1B model on
  hardware you already own does.
- **Swap the ghost.** Want Gemma, a bigger model, or a ghost in Spanish? Change `--model`.

### Where open beat closed

- **Small and specialised beats big and general here.** A frontier model knows more history, but for a tool that
  must answer *only* from your files, that extra knowledge is a liability. A 1B model trained for the job is a
  better fit, and it runs on anything.
- **One tiny client, many runtimes.** The open ecosystem settled on the same simple HTTP APIs, so one small
  `model.py` works with Ollama, llama.cpp, llamafile, LM Studio and vLLM.

## My Agent Session

<!-- TODO: link or embed the agent session -->

I built this together with Claude as a coding agent: picking the idea, writing the server and tests, and debugging
the fine-tuning notebook on Colab.

## Prize Categories

<!-- TODO: list any that apply, or remove this section -->

---

### Limitations

- Veronica only reads text. Images and binaries are served, but she can't read them.
- Keyword search matches words, not meaning, so "why did Gopher die?" matches "license" better than "killed".
- No `gophermap` support yet. Menus come from the folder layout.
- No TLS (very period-accurate), so don't serve anything private beyond your own network.

### Sources

[RFC 1436](https://www.rfc-editor.org/rfc/rfc1436) and the [Floodgap gopher server](gopher://gopher.floodgap.com/),
which has kept Gopherspace and Veronica-2 alive for years. The demo hole is my own summary; corrections welcome.

### License

The code is MIT. Unlike Gopher in 1993, you'll never be asked for a license fee.

The fine-tuned Veronica model is derived from Llama 3.2 1B and is covered by the
[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). **Built with Llama.**
