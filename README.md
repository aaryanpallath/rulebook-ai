# rulebook-ai — An NBA Rulebook Assistant, Measured

Ask a question about the official **2026-27 NBA Playing Rules** and get a short answer that cites the rule section
it comes from, or an honest "the rulebook does not address this". The project has three parts:

- **Retrieval** over the rulebook, split into 122 citable sections (196 chunks): keyword (BM25), local embeddings,
  and a hybrid of the two
- **An MCP server** (`search_rules`, `get_section`, `list_sections`), so Claude Code, Claude Desktop or any MCP client
  can use the rulebook as a tool
- **An evaluation harness**: 50 gold questions, five answering strategies, and accuracy, citation, abstention,
  latency and token-cost measurements

Everything runs locally: Gemma 4 (8B) and nomic-embed-text through Ollama on a 16 GB MacBook. Running the
evaluation costs nothing, and token counts show what a hosted model would cost.

## Results

42 answerable questions, phrased the way staff or fans ask them ("How long can a defender camp in the paint?"),
plus 8 the rulebook cannot answer (salary cap, draft lottery). Answers are graded automatically against facts that a
test checks appear in the cited rule text (`evals/questions.yaml`, every answer in `results/answers.jsonl`).

| Strategy | Correct (95% CI) | Cites the right section | Declines out-of-scope questions | Answers citing a section that doesn't exist | Median time | Tokens in / out |
|---|---|---|---|---|---|---|
| Model alone | 28.6% (17–44) | 0% | 5 / 8 | 4 | 0.9 s | 113 / 40 |
| RAG, hybrid search | 92.9% (81–97) | 81% | 8 / 8 | 0 | 5.3 s | 1,656 / 45 |
| **RAG, embedding search** | **95.2% (84–99)** | **88%** | **8 / 8** | **0** | 4.8 s | 1,584 / 44 |
| Agent v1: tools, short excerpts | 76.2% (62–87) | 83% | 8 / 8 | 0 | 4.8 s | 1,409 / 68 |
| Agent v2: full text, "search again" rule | 71.4% (56–83) | 60% | 8 / 8 | 0 | 5.5 s | 1,400 / 51 |
| Agent v3: search done before it starts | 90.5% (78–96) | 86% | 8 / 8 | 0 | 5.9 s | 1,742 / 52 |

**Retrieval alone** (is a correct section among the top results?):

| | Top 1 | Top 3 | Top 6 |
|---|---|---|---|
| BM25 keywords | 52% | 76% | 86% |
| Embeddings | 71% | **95%** | 95% |
| Hybrid (reciprocal rank fusion) | 67% | 86% | 95% |

### What I learned

- **Grounding is the whole game.** Alone, the model gets 29% right. It doesn't know 2026-27 changes, such as the
  second challenge awarded after a successful first one, and it invents citations to sections that don't exist.
  With the right sections in front of it, the same model is right 95% of the time and never cited a fake section.
- **Keywords lose to embeddings on how people actually ask.** "Get the ball past half court" never says
  "midcourt line" or "backcourt", so BM25 misses it. Fusing keywords with embeddings made things worse here:
  86% vs 95% at top 3.
- **An agent was not the right tool for this job.** Letting the model decide when to search scored below a fixed
  pipeline. Version 1 searched once, read short excerpts, and gave up on 8 answerable questions. Version 2 added
  an instruction to search again before giving up, which seemed obviously helpful, but the longer prompt made the
  model skip its tools entirely on 9 questions and scored lower. Version 3 runs the first search for it and
  reaches 90.5%. Since it almost never searches twice, it is effectively RAG with extra steps. Agents earn their
  keep on multi-step tasks; single-lookup questions want a pipeline.
- **A bug that only the agents had.** Every agent wrote the backboard as 6' by 3'; it's 3½', and RAG got it right.
  The tools returned JSON with Python's default `ensure_ascii`, so the model saw `3 \u00bd\u2019` instead of
  "3 ½’". With the fix and a test, agent v3 went from 88.1% to 90.5% and gets the backboard right. Versions 1 and
  2 ran with the bug.
- **Where it still misses.** Several strategies said the rulebook doesn't state the foul-out limit. The six-foul
  rule sits in Rule 3 (Team), not in the foul rules, so a search for "fouls" surfaces the wrong sections. That
  argues for adding glossary-style aliases to sections.

### Grading, and what changed

The grader checks required facts on word boundaries, so "4" doesn't match inside "14". It counts a citation as
correct if it names a gold section, and it counts an abstention as correct only on out-of-scope questions. While
reading results I fixed grader bugs, re-scored every strategy's saved answers the same way, and did not re-run
models:
- "twirl" vs "twirling" and "eject" vs "ejected" (whole-word matching)
- "after the third team foul" as a correct phrasing
- markdown bold breaking "will **not**"
- paraphrased refusals ("the rulebook does not contain...")

The three agent versions were developed against these same 50 questions, so a held-out question set is the next
step before trusting the 90.5%.

## Use it

```bash
uv sync
ollama pull gemma4 && ollama pull nomic-embed-text
uv run python -c "from rulebook_ai.ingest import build, fetch; build(fetch('data/rules.pdf'), 'data/sections.json')"
uv run python -m rulebook_ai.evaluate          # about 20 minutes on a laptop; resumes if interrupted
uv run python -m rulebook_ai.evaluate --regrade  # re-score saved answers without calling the model
uv run pytest                                  # parser, search, grader, gold-set integrity, MCP tools
```

**As an MCP server** (for example in Claude Code):

```bash
claude mcp add nba-rules -- uv --directory /path/to/rulebook-ai run rulebook-mcp
```

Then ask "What happens after a successful coach's challenge?" and the model searches the official text itself.

## How it's built

- `ingest.py`: downloads the official PDF and verifies its SHA-256. It splits the text into sections with stable
  ids: `R7-IV` is Rule 7, Section IV; Rule 12's parts become `R12A` and `R12B`; the Comments on the Rules become
  `C-II-L`. Long sections are split at their lettered items and keep the parent citation. Sub-items lettered I, V
  or X were first misread as new Roman-numeral parts, which a test now covers.
- `search.py`: BM25, cosine search over nomic-embed-text vectors, and reciprocal rank fusion.
- `answer.py`: model alone, RAG, and the three agent versions with `search_rules` and `get_section` tools. Every
  answer cites section ids in brackets.
- `mcp_server.py`: the same tools over the Model Context Protocol (Python SDK 2.x), tested with a real stdio client.
- `evaluate.py`: retrieval recall, graded answers, Wilson confidence intervals, latency and tokens. Answers are saved
  as they are produced: the first full run died when the local model exhausted 16 GB of memory at a 16k context,
  so the harness now retries, resumes, and runs at 8k.

## Limitations

- One 8B local model; a larger model would likely close the reading errors. The harness takes any Ollama model
  (`RULEBOOK_MODEL=...`).
- 50 questions give wide confidence intervals: 95% and 93% are not distinguishable, while 29% vs 95% clearly is.
- The rulebook PDF is the NBA's and is not redistributed here; the build downloads it from nba.com.

## License and AI disclosure

Code under MIT (see `LICENSE`). Built with help from Claude (Anthropic) via Claude Code; I reviewed, tested and
verified the code and results.
