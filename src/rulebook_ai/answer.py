"""Three ways to answer a rules question, compared in the evaluation:

closed_book  the model alone, from what it learned in training
rag          retrieve the top sections, then answer only from them
agent        the model drives search_rules / get_section tool calls itself, up to a step limit

Every answer must cite section ids in brackets, e.g. [R7-IV], or say the rulebook does not address the question."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .ingest import section_of
from .llm import chat
from .search import Index

ABSTAIN = "The rulebook does not address this."
SYSTEM = (
    "You answer questions about the official 2026-27 NBA playing rules for league staff. Answer in one to three "
    "sentences. Support every claim with the id of the rulebook section it comes from, in square brackets, e.g. "
    f"[R7-IV] or [C-II-L]. If the rulebook does not answer the question, reply exactly: {ABSTAIN}"
)
CITE = re.compile(r"\[((?:R\d+[AB]?|C)-[IVXL]+(?:-[A-Z])?)(?:#\d+)?\]")


@dataclass
class Answer:
    text: str
    citations: list[str]
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    steps: int = 1
    retrieved: list[str] = field(default_factory=list)


def _context(chunks) -> str:
    return "\n\n".join(f"[{section_of(c.id)}] {c.citation}\n{c.text}" for c in chunks)


def _finish(text: str, **kw) -> Answer:
    return Answer(text.strip(), sorted(set(CITE.findall(text))), **kw)


def closed_book(question: str) -> Answer:
    r = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}])
    return _finish(r.content, seconds=r.seconds, prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens)


def rag(question: str, index: Index, k: int = 6, method: str = "hybrid") -> Answer:
    chunks = index.search(question, k, method)
    prompt = f"Rulebook sections:\n\n{_context(chunks)}\n\nQuestion: {question}\nAnswer only from these sections."
    r = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return _finish(
        r.content,
        seconds=r.seconds,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        retrieved=[section_of(c.id) for c in chunks],
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": "Search the NBA rulebook. Returns the most relevant sections with ids.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "description": "What to look for, in plain words"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "Read the full text of one rulebook section by id, e.g. R14-V.",
            "parameters": {
                "type": "object",
                "required": ["section_id"],
                "properties": {"section_id": {"type": "string"}},
            },
        },
    },
]


def _json(obj: object) -> str:
    # keep "3 ½’" readable: with the default ensure_ascii the model saw "3 \u00bd\u2019" and read the size as 3'
    return json.dumps(obj, ensure_ascii=False)


def run_tool(index: Index, name: str, args: dict, full_text: bool = False) -> str:
    if name == "search_rules":
        hits = index.search(str(args.get("query", "")), 5)
        size = 4000 if full_text else 700  # version 1 saw 700-character excerpts and missed details further down
        return _json([{"id": section_of(h.id), "citation": h.citation, "text": h.text[:size]} for h in hits])
    if name == "get_section":
        parts = index.section_text(str(args.get("section_id", "")))
        if not parts:
            return _json({"error": "no such section"})
        return _json(
            {
                "id": section_of(parts[0].id),
                "citation": parts[0].citation,
                "text": " ".join(p.text for p in parts)[:6000],
            }
        )
    return _json({"error": f"unknown tool {name}"})


AGENT_V1 = " Use the tools to find the relevant sections before answering."
AGENT_V2 = (
    " Use the tools to find the relevant sections before answering. Search results contain the full text of"
    " each match; read it closely, including exceptions. If the first search does not answer the question,"
    " search again with different words (rulebook terms such as 'backcourt', 'disqualified', 'penalty') or"
    " read the whole section with get_section. Only say the rulebook does not address the question after at"
    " least two different searches."
)


def agent(question: str, index: Index, max_steps: int = 5, version: int = 3) -> Answer:
    """version 1: one-line instruction; search returns 700-character excerpts.
    version 2: full text plus a "search again before giving up" rule (the model then often skipped the tools).
    version 3: as 2, but the conversation starts with a search for the question already done, so the model always
    has evidence in front of it; it can still search again or read whole sections."""
    messages = [
        {"role": "system", "content": SYSTEM + (AGENT_V1 if version == 1 else AGENT_V2)},
        {"role": "user", "content": question},
    ]
    seconds = prompt = completion = 0.0
    retrieved: list[str] = []
    if version == 3:
        seeded = run_tool(index, "search_rules", {"query": question}, full_text=True)
        messages += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "search_rules", "arguments": {"query": question}}}],
            },
            {"role": "tool", "content": seeded, "tool_name": "search_rules"},
        ]
        retrieved += re.findall(r'"id": "([^"]+)"', seeded)
    for step in range(1, max_steps + 1):
        r = chat(messages, tools=TOOLS)
        seconds, prompt, completion = seconds + r.seconds, prompt + r.prompt_tokens, completion + r.completion_tokens
        if not r.tool_calls:
            return _finish(
                r.content,
                seconds=seconds,
                prompt_tokens=int(prompt),
                completion_tokens=int(completion),
                steps=step,
                retrieved=retrieved,
            )
        messages.append(r.message)
        for call in r.tool_calls:
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            result = run_tool(index, fn["name"], args, full_text=version >= 2)
            retrieved += [m for m in re.findall(r'"id": "([^"]+)"', result)]
            messages.append({"role": "tool", "content": result, "tool_name": fn["name"]})
    final = chat(messages + [{"role": "user", "content": "Answer now from what you found."}])
    return _finish(
        final.content,
        seconds=seconds + final.seconds,
        prompt_tokens=int(prompt + final.prompt_tokens),
        completion_tokens=int(completion + final.completion_tokens),
        steps=max_steps + 1,
        retrieved=retrieved,
    )
