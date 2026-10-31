# Fine-tune your own Veronica

Files here:

- `veronica_colab.ipynb`: the whole pipeline on a free Colab T4 GPU. A `qwen2.5:7b` teacher writes the lessons
  with `gophergeist dataset`, Unsloth LoRA-tunes `Llama-3.2-1B-Instruct`, and the result is exported as a 4-bit GGUF.
- `Modelfile`: loads that GGUF into Ollama as `llama3.2-veronica:1b`.

Open the notebook in Colab (File → Upload notebook), switch the runtime to a T4 GPU, set `REPO_URL`, and run
the cells top to bottom.

You can also build the dataset on any machine that runs a teacher model:

```bash
gophergeist dataset demo/hole ~/my-notes --model qwen2.5:7b --per-passage 8
```

That writes `veronica-train.jsonl` and `veronica-eval.jsonl` in the standard chat format
(`{"messages": [{"role": ..., "content": ...}]}`), which most fine-tuning tools read directly.

## Using Tinker instead

[Tinker](https://tinker-docs.thinkingmachines.ai/) reads this exact JSONL format through its cookbook's
`FromConversationFileBuilder`. Two things to know first:

- Its smallest current model is **Qwen3.5-4B**. The Llama 3.2 1B and 3B models are listed as retired. A 4B
  student is slower on a CPU than a 1B one, which works against the point of fine-tuning here.
- It's a paid API with no documented free tier, and it exports a Hugging Face model or LoRA adapter, not a GGUF.
  You'd convert it yourself with llama.cpp's `convert_hf_to_gguf.py` before loading it into Ollama.

For a fast CPU Veronica, the Colab notebook is the better fit.
