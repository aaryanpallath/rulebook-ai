"""Evaluate retrieval and answering on the gold questions.

    uv run python -m rulebook_ai.evaluate            # writes results/eval.json and results/answers.jsonl

Retrieval: does the gold section appear in the top k? (no LLM needed)
Answers, per mode:
  accuracy           answerable question: every required fact present and no abstention
  cited_gold         the answer cites at least one gold section
  invalid_citations  cited ids that do not exist in the rulebook (fabricated citations)
  abstained_correctly  unanswerable question: the model says the rulebook doesn't address it
  latency and tokens per question (a local model costs nothing to run; tokens show what a hosted one would cost)
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

import yaml

from . import answer
from .ingest import load, section_of
from .search import Index

ROOT = Path(__file__).resolve().parents[2]


def norm(t: str) -> str:
    """Lowercase, straighten quotes, drop markdown emphasis ("will **not**" -> "will not"), collapse spaces."""
    t = re.sub(r"(\*\*|__|\*|`)", "", t)
    return " ".join(t.lower().replace("’", "'").replace("”", '"').replace("“", '"').split())


def has(text: str, phrase: str) -> bool:
    """Phrase match on word boundaries, so '4' does not match inside '14'."""
    p = re.escape(norm(phrase))
    left = r"(?<![\w.])" if phrase[:1].isalnum() else ""
    right = r"(?![\w])" if phrase[-1:].isalnum() else ""
    return re.search(left + p + right, norm(text)) is not None


def abstained(text: str) -> bool:
    t = norm(text)
    return any(
        k in t
        for k in (
            "does not address",
            "doesn't address",
            "not addressed",
            "does not cover",
            "not covered",
            "no information",
            # paraphrased refusals ("the rulebook does not contain/specify ...") count as abstaining too
            "does not contain",
            "do not contain",
            "does not specify",
            "do not specify",
            "not specified",
            "does not explicitly state",
            "does not explicitly list",
            "not mentioned",
        )
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def grade(q: dict, a: answer.Answer, valid: set[str], answerable: bool) -> dict:
    row = {
        "id": q["id"],
        "answer": a.text,
        "citations": a.citations,
        "seconds": round(a.seconds, 2),
        "prompt_tokens": a.prompt_tokens,
        "completion_tokens": a.completion_tokens,
        "steps": a.steps,
        "invalid_citations": [c for c in a.citations if c not in valid],
    }
    if answerable:
        facts_ok = all(any(has(a.text, p) for p in group) for group in q["facts"])
        row.update(
            correct=facts_ok and not abstained(a.text),
            cited_gold=bool(set(a.citations) & set(q["gold"])),
            abstained=abstained(a.text),
        )
    else:
        row.update(correct=abstained(a.text), abstained=abstained(a.text))
    return row


def summarize(rows: list[dict], answerable_ids: set[str]) -> dict:
    ans = [r for r in rows if r["id"] in answerable_ids]
    un = [r for r in rows if r["id"] not in answerable_ids]
    k = sum(r["correct"] for r in ans)
    secs = [r["seconds"] for r in rows]
    return {
        "accuracy": round(k / len(ans), 3),
        "accuracy_ci95": wilson(k, len(ans)),
        "correct": k,
        "answerable": len(ans),
        "cited_gold": round(sum(r["cited_gold"] for r in ans) / len(ans), 3),
        "wrongly_abstained": sum(r["abstained"] for r in ans),
        "abstained_correctly": f"{sum(r['correct'] for r in un)}/{len(un)}",
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in rows),
        "median_seconds": round(statistics.median(secs), 2),
        "p90_seconds": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 2),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows)),
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in rows)),
        "mean_steps": round(statistics.mean(r["steps"] for r in rows), 2),
    }


def regrade() -> None:
    """Re-score saved answers with the current grader and gold set, without calling the model again."""
    questions = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text())
    by_id = {q["id"]: q for q in questions["answerable"] + questions["unanswerable"]}
    answerable_ids = {q["id"] for q in questions["answerable"]}
    valid = {section_of(s.id) for s in load(ROOT / "data" / "sections.json")}
    rows: dict[str, list[dict]] = {}
    lines = (ROOT / "results" / "answers.jsonl").read_text().splitlines()
    for line in lines:
        r = json.loads(line)
        a = answer.Answer(
            r["answer"], r["citations"], r["seconds"], r["prompt_tokens"], r["completion_tokens"], r["steps"]
        )
        rows.setdefault(r["mode"], []).append(
            {"mode": r["mode"], "question": r["question"], **grade(by_id[r["id"]], a, valid, r["id"] in answerable_ids)}
        )
    out = json.loads((ROOT / "results" / "eval.json").read_text())
    out["answers"] = {mode: summarize(rs, answerable_ids) for mode, rs in rows.items()}
    (ROOT / "results" / "eval.json").write_text(json.dumps(out, indent=2) + "\n")
    with open(ROOT / "results" / "answers.jsonl", "w") as f:
        for rs in rows.values():
            for r in rs:
                f.write(json.dumps(r) + "\n")
    for mode, s in out["answers"].items():
        print(mode, json.dumps(s))


def main() -> None:
    import sys

    if "--regrade" in sys.argv:
        return regrade()
    questions = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text())
    answerable, unanswerable = questions["answerable"], questions["unanswerable"]
    sections = load(ROOT / "data" / "sections.json")
    index = Index(sections, cache=ROOT / "data" / "embeddings.json")
    valid = {section_of(s.id) for s in sections}

    retrieval = {}
    for method in ("bm25", "embedding", "hybrid"):
        hits = {k: 0 for k in (1, 3, 6)}
        for q in answerable:
            got = [section_of(s.id) for s in index.search(q["q"], 6, method)]
            for k in hits:
                hits[k] += bool(set(got[:k]) & set(q["gold"]))
        retrieval[method] = {f"recall@{k}": round(v / len(answerable), 3) for k, v in hits.items()}
    print("retrieval", json.dumps(retrieval))

    modes = {
        "closed_book": answer.closed_book,
        "rag_hybrid": lambda q: answer.rag(q, index, method="hybrid"),
        "rag_embedding": lambda q: answer.rag(q, index, method="embedding"),
        "agent": lambda q: answer.agent(q, index, version=1),
        "agent_v2": lambda q: answer.agent(q, index, version=2),
        "agent_v3": lambda q: answer.agent(q, index, version=3),
    }
    answerable_ids = {q["id"] for q in answerable}
    # every graded answer is appended at once and a rerun skips what is already there, so a crash loses nothing
    (ROOT / "results").mkdir(exist_ok=True)
    log_path = ROOT / "results" / "answers.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            done[(r["mode"], r["id"])] = r
    results = {}
    with open(log_path, "a") as log:
        for name, fn in modes.items():
            rows = []
            t0 = time.monotonic()
            for q in answerable + unanswerable:
                if (name, q["id"]) in done:
                    rows.append(done[(name, q["id"])])
                    continue
                row = {"mode": name, "question": q["q"], **grade(q, fn(q["q"]), valid, q["id"] in answerable_ids)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                rows.append(row)
            results[name] = summarize(rows, answerable_ids)
            print(name, json.dumps(results[name]), f"({time.monotonic() - t0:.0f} s)", flush=True)

    from .llm import CHAT_MODEL, EMBED_MODEL

    out = {
        "model": {"chat": CHAT_MODEL, "embeddings": EMBED_MODEL, "temperature": 0, "context_tokens": 8192},
        "retrieval": retrieval,
        "answers": results,
    }
    (ROOT / "results" / "eval.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
