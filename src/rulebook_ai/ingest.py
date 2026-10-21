"""Turn the official NBA playing rules PDF into citable sections.

Each section of each rule becomes one retrievable unit with a stable id, e.g. "R7-II" (Rule 7, Section II) or
"R12B-IV" (Rule 12, Part B, Section IV); the Comments on the Rules become "C-II-B" and so on. Answers cite these ids,
so every claim can be checked against the rulebook text."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

RULES_URL = "https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2026/10/2026-27-Official-NBA-Playing-Rules.pdf"
RULES_SHA256 = "848c6f437be1299a36f48c53ccd9fdad2f3640d2449c27932a8f7ea806330a71"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130 Safari/537.36"


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


@dataclass
class Section:
    id: str
    rule: str  # "7", "12A", or "Comments"
    rule_title: str
    section: str
    title: str
    text: str
    page: int

    @property
    def citation(self) -> str:
        if self.rule == "Comments":
            return f"Comments on the Rules, {self.section} ({self.title})"
        return f"Rule {self.rule}, Section {self.section} ({self.title})"


def fetch(dest: str | Path) -> Path:
    """Download the 2026-27 rulebook (the NBA's copyright; not redistributed here) and verify it."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(RULES_URL, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
    if hashlib.sha256(dest.read_bytes()).hexdigest() != RULES_SHA256:
        raise ValueError(f"{dest} is not the expected 2026-27 rulebook (checksum mismatch)")
    return dest


def extract_lines(pdf_path: str | Path) -> list[tuple[int, str]]:
    import pdfplumber

    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            for line in (page.extract_text(x_tolerance=1.5) or "").splitlines():
                line = line.strip()
                if line and not re.fullmatch(r"-\s*\d+\s*-", line):  # page numbers
                    out.append((n, line))
    return out


RULE = re.compile(r"^RULE NO\.\s*(\d+)\s*[—-]\s*(.+)$")
PART = re.compile(r"^([AB])\.\s+(Technical Foul|Personal Foul)s?$")
SECTION = re.compile(r"^Section\s+([IVXL]+)\s*[—-]\s*(.+)$")
COMMENTS = re.compile(r"^COMMENTS ON THE RULES$")
C_TOP = re.compile(r"^([IVX]+)\.\s+([A-Z][A-Z ,'’/&—-]+)$")
C_SUB = re.compile(r"^([A-Z])\.\s+([A-Z][A-Z ,:'’/&—-]{3,})$")


def parse(lines: list[tuple[int, str]]) -> list[Section]:
    start = next(
        i for i, (_, ln) in enumerate(lines) if RULE.match(ln) and RULE.match(ln).group(1) == "1" and i > 50
    )  # skip the index at the front
    sections: list[Section] = []
    rule = rule_title = sec = title = ""
    body: list[str] = []
    page = 0
    in_comments = False
    c_top = ""

    def flush() -> None:
        if sec and body:
            sid = f"C-{sec}" if in_comments else f"R{rule}-{sec}"
            sections.append(
                Section(
                    sid,
                    "Comments" if in_comments else rule,
                    rule_title,
                    sec.replace("-", "(", 1) + (")" if "-" in sec else "") if in_comments else sec,
                    title,
                    " ".join(body),
                    page,
                )
            )

    for pg, ln in lines[start:]:
        if COMMENTS.match(ln):
            flush()
            in_comments, sec, body, rule_title = True, "", [], "Comments on the Rules"
            continue
        if not in_comments:
            if m := RULE.match(ln):
                flush()
                rule, rule_title, sec, body = m.group(1), m.group(2).title(), "", []
                continue
            if rule.startswith("12") and (m := PART.match(ln)):
                flush()
                rule, sec, body = "12" + m.group(1), "", []
                rule_title = f"Fouls and Penalties ({m.group(2)})"
                continue
            if m := SECTION.match(ln):
                flush()
                sec, title, body, page = m.group(1), m.group(2).strip(), [], pg
                continue
        else:
            # sub-items lettered I, V or X look like Roman numerals: only the next numeral in sequence is a new part
            expected = ROMAN[ROMAN.index(c_top) + 1] if c_top in ROMAN[:-1] else ("I" if not c_top else "")
            if (m := C_TOP.match(ln)) and m.group(1) == expected:
                flush()
                c_top, sec, title, body, page = m.group(1), m.group(1), m.group(2).title(), [], pg
                continue
            if (m := C_SUB.match(ln)) and c_top:
                flush()
                sec, title, body, page = f"{c_top}-{m.group(1)}", m.group(2).title(), [], pg
                continue
        if sec:
            body.append(ln)
    flush()
    return sections


MAX_CHARS = 1600


def split_long(sections: list[Section], max_chars: int = MAX_CHARS) -> list[Section]:
    """Split long sections at their lettered items (a., b., ...) into chunks of at most ~max_chars. Chunks keep the
    section's citation; their ids get a suffix (R13-I#2) so retrieval can point at the relevant part."""
    out = []
    for s in sections:
        if len(s.text) <= max_chars:
            out.append(s)
            continue
        items = re.split(r"(?=(?:^|\s)[a-z]\.\s)", s.text)
        chunks, cur = [], ""
        for it in items:
            if cur and len(cur) + len(it) > max_chars:
                chunks.append(cur.strip())
                cur = ""
            cur += it
            while len(cur) > max_chars * 1.5:  # an item that is itself very long: cut at a sentence
                cut = cur.rfind(". ", 0, max_chars) + 1 or max_chars
                chunks.append(cur[:cut].strip())
                cur = cur[cut:]
        if cur.strip():
            chunks.append(cur.strip())
        for i, c in enumerate(chunks, 1):
            out.append(Section(f"{s.id}#{i}", s.rule, s.rule_title, s.section, s.title, c, s.page))
    return out


def section_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def build(pdf_path: str | Path, out: str | Path) -> list[Section]:
    sections = split_long(parse(extract_lines(pdf_path)))
    Path(out).write_text(json.dumps([asdict(s) for s in sections], indent=1))
    return sections


def load(path: str | Path) -> list[Section]:
    return [Section(**d) for d in json.loads(Path(path).read_text())]
