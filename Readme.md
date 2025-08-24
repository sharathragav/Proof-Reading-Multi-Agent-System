# Proof-Reading Multi-Agent System

<p align="center">
  <img src="https://raw.githubusercontent.com/sharathragav/proof-reading-multi-agent-system/main/assets/logo.png" alt="logo" width="160" />
</p>

<p align="center">
  <a href="#"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/></a>
  <a href="#"><img alt="python" src="https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square"/></a>
  <a href="#"><img alt="chroma" src="https://img.shields.io/badge/vector_store-ChromaDB-lightgrey?style=flat-square"/></a>
  <a href="#"><img alt="langgraph" src="https://img.shields.io/badge/framework-LangGraph-purple?style=flat-square"/></a>
  <a href="#"><img alt="stars" src="https://img.shields.io/github/stars/sharathragav/proof-reading-multi-agent-system?style=social"/></a>
</p>

> **A multi-agent, RAG-enabled, style-aware proofreading system built with LangGraph.**
> Intelligent edits, auditable changelogs, human-in-the-loop review — optimized for speed and quality with a model cascade.

---

## 🔖 Table of Contents

* [What's Inside](#whats-inside)
* [Features](#features)
* [Architecture](#architecture)
* [Quickstart](#quickstart)
* [Installation](#installation)
* [Rule Management (RAG)](#rule-management-rag)
* [Usage](#usage)
* [Project Layout](#project-layout)
* [Style Rules Schema](#style-rules-schema)
* [Developer Notes & Debugging](#developer-notes--debugging)
* [Contributing](#contributing)
* [License & Contact](#license--contact)

---

## 🎯 What's Inside

A production-minded proofreading pipeline combining these ideas:

* **Retrieval-Augmented Generation (RAG)** for context-aware style application.
* **Model cascade**: fast small model for routine edits, escalate to large model for complex cases.
* **Auditable redlines** + machine-readable changelogs for traceability.
* **Human-in-the-loop**: Validator flags low-confidence or risky edits for manual review.

---

## ✨ Features

* ✅ Style-aware editing (US/UK/custom profiles)
* ⚡ Model cascade for speed + quality balance
* 🧠 Per-chunk retrieval from ChromaDB for style rules
* 📝 HTML redlines & `changelog.json` for every run
* 🔎 Validator that classifies changes and assigns confidence
* ♻️ Modular: swap vector store or models with minimal changes
* 📦 CLI-first with `pipeline` and `native` modes

---

## 🏗 Architecture

Workflows are implemented as a **stateful LangGraph** pipeline of focused nodes:

1. **Document Processor** — ingest & chunk via `docling`.
2. **Retriever** — query ChromaDB for top-matching style rules per-chunk.
3. **Style Knowledge Builder** — assemble instruction + examples for the model.
4. **Neural Edit Agent** — attempt edit with small model; escalate if needed.
5. **Validator** — diff + classify + confidence score.
6. **Router** — auto-approve or queue for human review.
7. **Changelog & Redline Generator** — produce artifacts per-run.

This keeps processing transparent, reproducible, and debuggable.

---

## 🚀 Quickstart (TL;DR)

```bash
git clone https://github.com/sharathragav/proof-reading-multi-agent-system.git
cd proof-reading-multi-agent-system
python -m venv .venv
# Windows
.venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/index_rules.py path/to/rules.json
python main.py
```

> Tip: Start with `native` mode for a quick check, then run `pipeline` for full RAG + validator checks.

---

## ⚙️ Installation

**Recommended**: Python 3.12+, CUDA-enabled GPU for local model runs. If you don't have CUDA, use CPU-only PyTorch or a hosted LLM.

Install PyTorch (example for CUDA 11.8):

```bash
pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Download SpaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## 📚 Rule Management (RAG)

Style rules are first-class — store them in JSON, index into ChromaDB, and the system will query them per-document chunk.

**Flow:**

1. Prepare `rules.json` (array of rule objects — see schema below).
2. `python scripts/index_rules.py path/to/rules.json` → embeds rules into `./chroma_db`.
3. Run `python main.py` and select your style profile.

**Why RAG?** You can continuously evolve your style knowledge base without changing code. Update rules, re-index, and the editor learns new guidance instantly.

---

## ▶️ Usage

Run the CLI and follow prompts:

```bash
python main.py
```

Example prompts:

* `Enter the path to the document file:` → `./docs/sample.docx`
* `Enter style profile (us/uk) [default: us]:` → `us`
* `Select execution mode - 'native' or 'pipeline' [default: pipeline]:` → `pipeline`

**Outputs (per-run)**

* `./artifacts/<run-timestamp>/changelog.json` — structured edits & metadata
* `./artifacts/<run-timestamp>/redline.html` — visual diff
* `./artifacts/<run-timestamp>/logs/` — validator and runtime logs

---

## 📁 Project Layout

```
proof-reading-multi-agent-system/
├── main.py                 # CLI entrypoint
├── requirements.txt
├── scripts/                # helpers: extract & index rules
│   ├── extract_rules_from_docx.py
│   ├── extract_rules_from_pdf.py
│   └── index_rules.py
├── src/                    # core code
│   ├── langgraph_pipeline.py
│   ├── neural_edit_agent.py
│   ├── change_tracker.py
│   ├── document_processor.py
│   ├── vector_client.py
│   ├── model_loader.py
│   └── ...
├── artifacts/              # generated artifacts per run
└── chroma_db/              # local vector DB
```

---

## 🧾 Style Rules JSON Schema (example)

```json
{
  "rule_id": "spelling-british-vs-american",
  "content": "Use British spelling for words such as 'colour', 'organise' when profile == 'uk'.",
  "priority": 10,
  "tags": ["spelling", "uk"],
  "examples": [
    {"before": "color", "after": "colour"}
  ]
}
```

**Fields:**

* `rule_id` — unique id
* `content` — instruction text
* `priority` — higher = stronger precedence
* `tags` — search & filter
* `examples` — optional before/after pairs for prompt examples

---

## 🛠 Developer Notes & Debugging

* **Unborn HEAD**: Make the initial commit locally before pushing to remote (classic `git` gotcha).
* **Model cascade tuning**: Start conservative; iterate with validator reports.
* **Validator**: Flags numeric and named-entity edits as high-risk by default.
* **Swap vector store**: `vector_client.py` is modular — swap Chroma with minimal changes.

**Debugging tips**:

* Inspect `artifacts/<timestamp>/validator_report.json` for diffs, change classification, and confidence.
* Use `document_processor` functions to extract problematic chunk text and feed directly to `neural_edit_agent.py`.

---

## 🤝 Contributing

Contributions welcome. Steps:

1. Fork the repo
2. Create a feature branch
3. Add tests where possible
4. Open a PR with a clear description and screenshots for UI changes

## 📜 License & Contact

Released under the **MIT License**. See `LICENSE`.

Maintainer: **Sharath Ragav** — feel free to open issues or PRs.

---

<p align="center">Made with ❤️ using LangGraph — built for audits, quality, and real-world proofreading.</p>

*Want a condensed `README-short.md` for PyPI or a formatted HTML landing page? Tell me and I’ll add it and propose a commit.*
