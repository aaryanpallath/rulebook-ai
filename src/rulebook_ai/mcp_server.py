"""An MCP server exposing the rulebook to any MCP client (Claude Code, Claude Desktop, Copilot Studio, ...):

    claude mcp add nba-rules -- uv --directory /path/to/rulebook-ai run rulebook-mcp

Tools: search_rules (hybrid search), get_section (full text by id), list_sections (the table of contents).
The model brings the reasoning; the server guarantees the text it cites is the official rulebook's."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .ingest import build, fetch, load, section_of
from .search import Index

DATA = Path(os.environ.get("RULEBOOK_DATA", Path(__file__).resolve().parents[2] / "data"))
mcp = MCPServer("nba-rules", instructions="Official 2026-27 NBA playing rules. Cite section ids from these tools.")


@cache
def index() -> Index:
    sections_file = DATA / "sections.json"
    if not sections_file.exists():
        build(fetch(DATA / "rules.pdf"), sections_file)
    use_embeddings = os.environ.get("RULEBOOK_EMBEDDINGS", "1") == "1"
    return Index(load(sections_file), cache=DATA / "embeddings.json", use_embeddings=use_embeddings)


@mcp.tool()
def search_rules(query: str, k: int = 5) -> list[dict]:
    """Search the 2026-27 NBA playing rules. Returns the most relevant sections with ids and an excerpt."""
    return [
        {"id": section_of(s.id), "citation": s.citation, "page": s.page, "excerpt": s.text[:700]}
        for s in index().search(query, max(1, min(k, 10)))
    ]


@mcp.tool()
def get_section(section_id: str) -> dict:
    """Full text of one rulebook section by id, e.g. 'R14-V' (Rule 14, Section V) or 'C-II-L' (Comments)."""
    parts = index().section_text(section_id)
    if not parts:
        return {"error": f"no section {section_id}; use list_sections to see valid ids"}
    return {
        "id": section_of(parts[0].id),
        "citation": parts[0].citation,
        "page": parts[0].page,
        "text": " ".join(p.text for p in parts),
    }


@mcp.tool()
def list_sections() -> list[dict]:
    """The rulebook's table of contents: every section id with its title."""
    seen, out = set(), []
    for s in index().sections:
        sid = section_of(s.id)
        if sid not in seen:
            seen.add(sid)
            out.append({"id": sid, "citation": s.citation})
    return out


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
