# Automating ML/DL Paper-to-Code Translation via Multi-Agent LLM Pipelines

This repository contains an agentic GenAI system that translates ML/DL research papers
into runnable, verifiable codebases. The initial scope targets **NLP (Transformers)** and **CV (CNN/ViT)**,
plus one classic ML baseline (e.g., XGBoost).

## Week 1 Status
- [x] Project scaffold created
- [x] Core folders laid out
- [x] Base tooling defined

## Project Structure
```
.
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
│   ├── parsing/
│   ├── ir_generation/
│   └── code_generation/
├── agents/
│   ├── parser_agent/
│   ├── mapping_agent/
│   └── verification_agent/
├── models/
├── references/
│   └── papers/    # Store PDFs locally; avoid committing large files
├── tests/
└── requirements.txt
```

## Selected Papers (initial)
**NLP**
- Vaswani et al., 2017 — "Attention Is All You Need" (Transformers)
- A modern Transformer survey/tutorial (placeholder)

**CV**
- He et al., 2015 — "Deep Residual Learning for Image Recognition" (ResNet)
- Dosovitskiy et al., 2020 — "An Image is Worth 16×16 Words" (ViT)

**Classic ML**
- Chen & Guestrin, 2016 — "XGBoost: A Scalable Tree Boosting System"

> Tip: Keep PDFs in `references/papers/` but **do not commit** large files. Use links in docs or Git LFS if needed.

## Getting Started (local)
```bash
python3 -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## License
MIT
