import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from rulebook_ai.answer import CITE, run_tool
from rulebook_ai.evaluate import abstained, has, wilson
from rulebook_ai.ingest import Section, load, parse, section_of, split_long
from rulebook_ai.search import BM25, Index, tokens

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "data" / "sections.json"

LINES = [
    (1, "RULES INDEX"),
    *[(1, "filler")] * 60,
    (9, "RULE NO. 1—COURT DIMENSIONS—EQUIPMENT"),
    (9, "Section I—Court and Dimensions"),
    (9, "a. The free throw line is 15’ from the backboard."),
    (39, "RULE NO. 12—FOULS AND PENALTIES"),
    (39, "A. Technical Foul"),
    (39, "Section I—Excessive Timeouts"),
    (39, "a. A technical foul shall be assessed."),
    (44, "B. Personal Foul"),
    (44, "Section I—Types"),
    (44, "a. A player shall not hold an opponent."),
    (65, "COMMENTS ON THE RULES"),
    (65, "I. GUIDES FOR ADMINISTRATION"),
    (65, "Officials should be uniform."),
    (66, "II. BASIC PRINCIPLES"),
    (66, "A. CONTACT SITUATIONS"),
    (66, "Contact is not always a foul."),
    (67, "I. OFFENSIVE THREE-SECONDS"),
    (67, "The count starts in the frontcourt."),
    (68, "V. VERBAL FAN INTERFERENCE"),
    (68, "Fans may not distract players."),
]


def test_parser_builds_rule_part_section_and_comment_ids():
    ids = [s.id for s in parse(LINES)]
    # "I." and "V." inside Comments II are sub-items, not new top-level parts
    assert ids == ["R1-I", "R12A-I", "R12B-I", "C-I", "C-II-A", "C-II-I", "C-II-V"]
    s = parse(LINES)[0]
    assert s.citation == "Rule 1, Section I (Court and Dimensions)" and s.page == 9
    assert parse(LINES)[4].citation == "Comments on the Rules, II(A) (Contact Situations)"


def test_long_sections_split_at_items_and_keep_their_citation():
    long = Section(
        "R7-IV", "7", "Shot Clock", "IV", "Resetting", " ".join(f"{c}. " + "word " * 120 for c in "abcdef"), 28
    )
    parts = split_long([long], max_chars=800)
    assert len(parts) > 1 and all(len(p.text) <= 1200 for p in parts)
    assert [section_of(p.id) for p in parts] == ["R7-IV"] * len(parts)
    assert {p.citation for p in parts} == {long.citation}
    assert " ".join(p.text for p in parts).split() == long.text.split()


def test_bm25_and_tokens():
    assert tokens("The players dribbling the balls") == ["player", "dribbl", "ball"]
    bm = BM25(["shot clock reset to fourteen", "free throw shooter has ten seconds", "court dimensions"])
    assert bm.scores("how long does the free throw shooter have").argmax() == 1


def test_citation_parsing_and_grading():
    text = "Reset to 14 [R7-IV#2]; see also [C-II-L] and [R12B-XI]. Not a citation: [R99]"
    assert sorted(CITE.findall(text)) == ["C-II-L", "R12B-XI", "R7-IV"]
    assert has("The shot clock resets to 14 seconds", "14") and not has("resets to 24 seconds", "4")
    assert has("at least :00.3 must expire", ":00.3")
    assert abstained("The rulebook does not address this.") and not abstained("It is 24 seconds [R7-I].")
    lo, hi = wilson(30, 42)
    assert lo < 30 / 42 < hi


@pytest.mark.skipif(not SECTIONS.exists(), reason="run the ingest first (it downloads the rulebook)")
def test_gold_answers_are_in_the_cited_sections():
    sections = load(SECTIONS)
    text: dict[str, list[str]] = {}
    for s in sections:
        text.setdefault(section_of(s.id), []).append(s.text)
    q = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text())
    assert len(q["answerable"]) == 42 and len(q["unanswerable"]) == 8
    for item in q["answerable"]:
        assert all(g in text for g in item["gold"]), item["id"]
        if not item.get("inferred"):
            source = " ".join(t for g in item["gold"] for t in text[g])
            for group in item["facts"]:
                assert any(has(source, p) for p in group), (item["id"], group)


@pytest.mark.skipif(not SECTIONS.exists(), reason="run the ingest first")
def test_tools_and_mcp_server_without_embeddings():
    os.environ["RULEBOOK_EMBEDDINGS"] = "0"
    from rulebook_ai import mcp_server

    index = Index(load(SECTIONS), use_embeddings=False)
    hits = json.loads(run_tool(index, "search_rules", {"query": "coach challenge timeout"}))
    assert hits and all(h["id"].startswith(("R", "C")) for h in hits)
    full = json.loads(run_tool(index, "get_section", {"section_id": "R14-I#1"}))
    assert full["id"] == "R14-I" and "second Challenge" in full["text"]
    assert "error" in json.loads(run_tool(index, "get_section", {"section_id": "R99-I"}))
    raw = run_tool(index, "get_section", {"section_id": "R1-II"})
    assert "3 ½" in raw and "\\u00bd" not in raw  # the model must see the character, not an escape

    tools = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert tools == {"search_rules", "get_section", "list_sections"}
    assert mcp_server.get_section("R5-II")["citation"] == "Rule 5, Section II (Timing)"
    assert len(mcp_server.list_sections()) == 122
