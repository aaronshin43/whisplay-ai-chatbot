"""Convert Red Cross Wilderness & Remote First Aid ERG PDF to Markdown."""
import pdfplumber
import re
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF_PATH   = "D:/GitHub/whisplay-ai-chatbot-1/wilderness-remote-first-aid-erg.pdf"
OUTPUT_PATH = "D:/GitHub/whisplay-ai-chatbot-1/data/knowledge/redcross_wilderness.md"

# ── Noise patterns to strip ────────────────────────────────────────────────
NOISE_EXACT = {
    "Wilderness and Remote First Aid Emergency Reference Guide",
    "wilderness and remote first aid emergency reference guide",
    "Wilderness and Remote First Aid",
    "injuries and illnesses",
    "Photo Credits",
    "Contents",
    "Acknowledgments",
    "SpeCiAl SiTuATioNS",
}

NOISE_RE = [
    re.compile(r"^\d{1,3}\s*(wilderness and remote first aid.*)?$", re.I),
    re.compile(r"^injuries and illnesses\s*\d*$", re.I),
    re.compile(r"^[ivxlcdm]+$", re.I),     # roman numerals
    re.compile(r"^[.\-_\s]{4,}$"),          # dot/dash filler lines
    re.compile(r"^©.*$"),                   # copyright
    re.compile(r"^ISBN.*$"),
]

# Lines to keep but reformat as section heading
HEADING_TRIGGERS = {
    "Check Call Care", "CHECK CALL CARE",
}


def is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s in NOISE_EXACT:
        return True
    for pat in NOISE_RE:
        if pat.fullmatch(s):
            return True
    # pure page-number lines like "13", "14 wilderness..."
    if re.match(r"^\d{1,3}\s*$", s):
        return True
    return False


def classify_line(line: str) -> tuple[str, str]:
    """
    Returns (kind, text) where kind is:
      'h2'  - major section heading
      'h3'  - sub-heading
      'li'  - list item (bullet)
      'p'   - normal paragraph text
    """
    s = line.strip()

    # Explicit bullets (n <text>, • <text>, - <text>)
    if re.match(r"^[n•\-]\s+\S", s):
        body = re.sub(r"^[n•\-]\s+", "", s)
        return ("li", body)

    # Numbered steps
    if re.match(r"^\d+\s+[A-Z]", s):
        return ("li", s)

    # ALL-CAPS ≤ 8 words, no period → h2
    if s.isupper() and 1 <= len(s.split()) <= 8 and not s.endswith("."):
        return ("h2", s.title())

    # Title-case ≤ 7 words, starts upper, no period → h3
    words = s.split()
    if (
        s[0].isupper()
        and 2 <= len(words) <= 7
        and not s.endswith(".")
        and not s.endswith(",")
        and not s[0].isdigit()
        and sum(1 for c in s if c.isupper()) >= 2
        and len(s) <= 65
    ):
        return ("h3", s)

    return ("p", s)


def extract_all(pdf) -> list[str]:
    """Extract raw non-noise lines from all pages, skipping index (p116+)."""
    raw = []
    for i, page in enumerate(pdf.pages):
        if i >= 115:   # stop before index section (page 116+)
            break
        text = page.extract_text() or ""
        for line in text.split("\n"):
            if not is_noise(line):
                raw.append(line.strip())
        raw.append("")   # page separator
    return raw


def build_markdown(raw_lines: list[str]) -> str:
    md_parts = []
    prev_blank = True

    for line in raw_lines:
        s = line.strip()

        if not s:
            if not prev_blank:
                md_parts.append("")
            prev_blank = True
            continue

        prev_blank = False
        kind, text = classify_line(s)

        if kind == "h2":
            md_parts.append(f"\n## {text}")
        elif kind == "h3":
            md_parts.append(f"\n### {text}")
        elif kind == "li":
            md_parts.append(f"- {text}")
        else:
            md_parts.append(text)

    # Collapse 3+ blank lines → 2
    result = []
    blanks = 0
    for line in md_parts:
        if line == "":
            blanks += 1
            if blanks <= 2:
                result.append("")
        else:
            blanks = 0
            result.append(line)

    return "\n".join(result)


HEADER = """\
# Wilderness and Remote First Aid — Emergency Reference Guide

**Source:** American Red Cross — Wilderness and Remote First Aid Emergency Reference Guide
**Standard:** American Red Cross / SOLO Wilderness Medicine
**Category:** Wilderness & Remote First Aid — Environmental Emergencies and Field Protocols
**Applies to:** All ages; field settings without rapid EMS access

[DOMAIN_TAGS: wilderness, remote_first_aid, snakebite, envenomation, hypothermia, frostbite, heat_exhaustion, heat_stroke, hyperthermia, lightning, avalanche, altitude_sickness, drowning, dehydration, wound_infection, blister, fracture, dislocation, spinal_injury, CPR, AED, choking, shock, anaphylaxis, seizure, stroke, asthma, burns, eye_injury, poisoning, evacuation, carry_techniques, rescue, improvised_splint, bleeding_control, tourniquet, check_call_care]

---
"""


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"PDF loaded: {len(pdf.pages)} pages", flush=True)
        raw_lines = extract_all(pdf)

    print(f"Raw lines extracted: {len(raw_lines)}", flush=True)

    body = build_markdown(raw_lines)
    full_doc = HEADER + body

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_doc)

    line_count = full_doc.count("\n")
    size_kb = len(full_doc.encode("utf-8")) / 1024
    print(f"Written: {OUTPUT_PATH}")
    print(f"  {line_count} lines, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
