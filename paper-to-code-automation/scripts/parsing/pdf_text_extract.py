import os
import json
import pdfplumber
import re
from tqdm import tqdm

HEADERS = [
    "abstract", "introduction", "background", "related work",
    "method", "methods", "approach", "proposed method", "algorithm",
    "experiments", "results", "evaluation", "discussion",
    "conclusion", "future work", "acknowledgments", "references"
]

def normalize(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

def find_title(first_page_text: str) -> str:
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    title_lines = []
    for line in lines:
        if re.match(r"(?i)abstract", line):
            break
        title_lines.append(line)
        if len(title_lines) >= 3:
            break
    return normalize(" ".join(title_lines))

def split_sections(full_text: str):
    pattern = r"(?im)^\s*(\d+(\.\d+)*)?\s*(" + "|".join([re.escape(h) for h in HEADERS]) + r")\s*$"
    sections = {}
    current = "preamble"
    sections[current] = []
    for line in full_text.splitlines():
        if re.match(pattern, line.strip()):
            current = re.sub(pattern, lambda m: m.group(3).lower(), line.strip())
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    for k in list(sections.keys()):
        sections[k] = normalize("\n".join(sections[k]))
    return sections

def extract_pseudocode_blocks(full_text: str):
    blocks = []
    alg_markers = re.finditer(r"(?is)(algorithm\s*\d*[:\.\)]?.{0,120})", full_text)
    for m in alg_markers:
        start = max(0, m.start()-400)
        end = min(len(full_text), m.end()+600)
        snippet = full_text[start:end]
        blocks.append(normalize(snippet))
    return blocks

def extract_pdf_text(pdf_path: str):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting text"):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            texts.append(txt)
    full_text = "\n".join(texts)
    first_page = texts[0] if texts else ""
    title = find_title(first_page)
    sections = split_sections(full_text)
    pseudocode = extract_pseudocode_blocks(full_text)
    abstract = ""
    for k in sections:
        if k.lower() == "abstract":
            abstract = sections[k]
            break
    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "pseudocode_blocks": pseudocode,
        "raw_text_length": len(full_text)
    }