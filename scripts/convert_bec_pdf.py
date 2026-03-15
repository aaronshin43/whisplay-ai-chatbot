"""Convert WHO BEC PDF to per-module Markdown files."""
import pdfplumber
import re
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF_PATH = "D:/GitHub/whisplay-ai-chatbot-1/BASIC EMERGENCY CARE_WHO.pdf"
OUTPUT_DIR = "D:/GitHub/whisplay-ai-chatbot-1/data/knowledge"

# Sidebar navigation tab labels printed as standalone lines
NAV_WORDS = {
    "INTRO", "ABCDE", "TRAUMA", "BREATHING", "SHOCK", "AMS", "SKILLS",
    "GLOSSARY", "REFS", "&", "QUICK", "CARDS",
    "PARTICIPANT WORKBOOK", "WORKBOOK", "PARTICIPANT",
    "BASIC EMERGENCY CARE", "APPROACH TO THE ACUTELY ILL AND INJURED",
    "Basic emergency care",
}

# Page ranges (1-indexed, inclusive)
SECTIONS = {
    "module1":     (20, 44),
    "module2":     (46, 82),
    "module3":     (84, 102),
    "module4":     (104, 126),
    "module5":     (128, 148),
    "skills":      (152, 210),
    "quick_cards": (225, 238),
}

CONFIGS = {
    "who_bec_module1_abcde": {
        "section": "module1",
        "title": "# WHO BEC Module 1: The ABCDE and SAMPLE History Approach",
        "tags": "[DOMAIN_TAGS: ABCDE, primary_survey, airway, breathing, circulation, disability, exposure, SAMPLE_history, vital_signs, triage, assessment, unconscious, responsiveness]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Primary Survey — ABCDE Assessment Framework\n"
        ),
    },
    "who_bec_module2_trauma": {
        "section": "module2",
        "title": "# WHO BEC Module 2: Approach to Trauma",
        "tags": "[DOMAIN_TAGS: trauma, hemorrhage, bleeding, fracture, head_injury, spinal_injury, burns, wound_care, tourniquet, C-spine, immobilization, penetrating_trauma, blunt_trauma]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Trauma — Hemorrhage, Fractures, Head Injury, Burns\n"
        ),
    },
    "who_bec_module3_breathing": {
        "section": "module3",
        "title": "# WHO BEC Module 3: Approach to Difficulty in Breathing",
        "tags": "[DOMAIN_TAGS: breathing, dyspnea, asthma, pneumonia, tension_pneumothorax, COPD, respiratory_failure, oxygen_therapy, nebulizer, chest_decompression, wheeze, stridor]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Respiratory Emergency — Dyspnea, Asthma, Pneumonia, Tension Pneumothorax\n"
        ),
    },
    "who_bec_module4_shock": {
        "section": "module4",
        "title": "# WHO BEC Module 4: Approach to Shock",
        "tags": "[DOMAIN_TAGS: shock, hypovolemic_shock, septic_shock, anaphylactic_shock, cardiogenic_shock, IV_fluids, blood_pressure, perfusion, vasopressors, fluid_resuscitation, tachycardia, hypotension]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Shock — Hypovolemic, Distributive, Obstructive, Cardiogenic\n"
        ),
    },
    "who_bec_module5_mental_status": {
        "section": "module5",
        "title": "# WHO BEC Module 5: Approach to Altered Mental Status",
        "tags": "[DOMAIN_TAGS: altered_mental_status, AMS, seizure, stroke, hypoglycemia, meningitis, GCS, AVPU, coma, encephalopathy, intracranial_pressure, confusion, unresponsive]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Neurological Emergency — AMS, Seizure, Stroke, Hypoglycemia, Meningitis\n"
        ),
    },
    "who_bec_skills": {
        "section": "skills",
        "title": "# WHO BEC Skills Reference",
        "tags": "[DOMAIN_TAGS: skills, airway_management, IV_access, oxygen_delivery, wound_care, splinting, CPR, defibrillation, needle_decompression, bag_valve_mask, suction, nasopharyngeal_airway, oropharyngeal_airway, chest_compression, rescue_breathing]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook, Skills Section\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Clinical Skills — Procedures and Techniques\n"
        ),
    },
    "who_bec_quick_cards": {
        "section": "quick_cards",
        "title": "# WHO BEC Quick Reference Cards",
        "tags": "[DOMAIN_TAGS: quick_reference, critical_actions, drug_dosages, medications, ABCDE, trauma_protocol, shock_protocol, AMS_protocol, pediatric, handover, transfer, epinephrine, dextrose]",
        "meta": (
            "**Source:** WHO Basic Emergency Care (BEC) — Participant Workbook, Quick Cards\n"
            "**Standard:** WHO / ICRC 2018\n"
            "**Category:** Quick Reference — Critical Actions and Drug Dosages\n"
        ),
    },
}


def clean_line(line: str):
    s = line.strip()
    if not s:
        return None
    # Nav/tab sidebar words
    if s in NAV_WORDS:
        return None
    # Standalone page numbers
    if re.fullmatch(r"\d{1,3}", s):
        return None
    # Pure dot filler lines
    if re.fullmatch(r"[.\s]+", s):
        return None
    # Note dotted lines (.....)
    if re.fullmatch(r"[.\-_\s]{5,}", s):
        return None
    return s


def extract_pages(pdf, start_page: int, end_page: int) -> list[str]:
    lines_out = []
    for page_idx in range(start_page - 1, end_page):
        page = pdf.pages[page_idx]
        text = page.extract_text() or ""
        for line in text.split("\n"):
            cleaned = clean_line(line)
            if cleaned is not None:
                lines_out.append(cleaned)
        lines_out.append("")  # inter-page gap
    return lines_out


def lines_to_markdown(lines: list[str]) -> str:
    md = []
    prev_blank = True

    for line in lines:
        if not line.strip():
            if not prev_blank:
                md.append("")
            prev_blank = True
            continue
        prev_blank = False

        s = line.strip()

        # Already looks like a markdown heading — pass through
        if s.startswith("#"):
            md.append(s)
            continue

        words = s.split()
        word_count = len(words)

        # ALL-CAPS short phrase (≤8 words, ≤80 chars, no period) → ## heading
        if (
            s.isupper()
            and word_count <= 8
            and len(s) <= 80
            and not s.endswith(".")
        ):
            md.append(f"## {s.title()}")
            continue

        # Numbered list items — preserve
        if re.match(r"^\d+\.\s", s) or re.match(r"^[a-z]\.\s", s):
            md.append(s)
            continue

        # Bullet/dash list items — preserve
        if s.startswith(("- ", "* ", "• ")):
            md.append(s)
            continue

        # Title-ish line: starts uppercase, ≤7 words, no trailing period, mixed case OK
        if (
            s[0].isupper()
            and word_count <= 7
            and len(s) <= 70
            and not s.endswith(".")
            and not s.endswith(",")
            and sum(1 for c in s if c.isupper()) >= 2
        ):
            md.append(f"### {s}")
            continue

        md.append(s)

    # Collapse excessive blank lines
    result = []
    blank_count = 0
    for line in md:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result.append("")
        else:
            blank_count = 0
            result.append(line)

    return "\n".join(result)


def build_doc(cfg: dict, raw_lines: list[str]) -> str:
    header = (
        cfg["title"] + "\n\n"
        + cfg["meta"] + "\n"
        + cfg["tags"] + "\n\n"
        + "---\n\n"
    )
    body = lines_to_markdown(raw_lines)
    return header + body


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with pdfplumber.open(PDF_PATH) as pdf:
        total = len(pdf.pages)
        print(f"PDF loaded: {total} pages")

        for filename, cfg in CONFIGS.items():
            start, end = SECTIONS[cfg["section"]]
            print(f"Extracting {filename} (pages {start}-{end})...", flush=True)
            raw_lines = extract_pages(pdf, start, end)
            doc = build_doc(cfg, raw_lines)
            out_path = os.path.join(OUTPUT_DIR, filename + ".md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(doc)
            line_count = doc.count("\n")
            size = len(doc.encode("utf-8"))
            print(f"  -> {filename}.md  ({line_count} lines, {size:,} bytes)")

    print("\nAll done!")


if __name__ == "__main__":
    main()
