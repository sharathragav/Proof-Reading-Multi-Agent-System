# Proofreader System

A multi-agent, RAG-enabled, style-aware proofreading service built using the LangGraph framework.

## Features

- Document ingestion using Docling
- Style-aware editing with RAG-enabled rule retrieval
- Model cascade (Flan-T5-small → coedit-large)
- Human-in-the-loop (HITL) review system
- Auditable changelogs and HTML redlines
- Support for multiple style guides (US/UK English)

## Quick Start

1. Install dependencies:
   ```bash
   py -3.12 -m venv .venv
   .\.venv\Scripts\activate
   pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m spacy validate
   uv pip freeze > uv.lock
   pip install -r requirements.lock


## Project Structure

proofreader/
├── README.md
├── requirements.txt
├── rules/
│   └── rules.json
├── scripts/
│   └── index_rules_weaviate.py
├── src/
│   ├── __init__.py
│   ├── langgraph_pipeline.py
│   ├── document_processor.py
│   ├── vector_client.py
│   ├── model_loader.py
│   ├── neural_edit_agent.py
│   ├── proofreader.py
│   ├── change_tracker.py
│   └── utils.py
├── artifacts/         # runtime outputs (changelogs, redlines)
└── checkpoints/       # pipeline checkpoints (MemorySaver)



