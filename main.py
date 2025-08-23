import argparse
import asyncio
import os
from src.langgraph_pipeline import process_document_with_graph
from src.model_loader import load_models
from src.neural_edit_agent import NeuralEditAgent
from src.document_processor import process_document



def run_native(document_path, style_profile):
    print("[Native Mode] Proofreading with a single model...")
    # Load models
    small_tok, small_model, small_dev, _, _, _ = load_models()
    agent = NeuralEditAgent(small_tok, small_model, small_dev, small_tok, small_model, small_dev)
    doc = process_document(document_path)
    chunks = doc.get("chunks", [])
    for i, chunk in enumerate(chunks):
        edited, meta = agent.edit_chunk(
            chunk["text"],
            instruction_block=f"Apply {style_profile.upper()} English style.",
            exemplars=None,
            post_rules=None,
            protected_mask=chunk.get("protected_spans", [])
        )
        print(f"\n--- Chunk {i} ---\nOriginal:\n{chunk['text']}\n\nEdited:\n{edited}\n")


def run_pipeline(document_path, style_profile):
    print("[Pipeline Mode] Running LangGraph multi-agent pipeline...")
    result = asyncio.run(process_document_with_graph(document_path, style_profile))
    print(f"\nJob completed: {result['job_id']}")
    print(f"Processed {len(result['chunks'])} chunks")
    print(f"Changes made: {len(result.get('changes', []))}")
    print(f"Chunks needing review: {len(result.get('human_queue', []))}")
    print(f"Artifacts saved in ./artifacts/")


def main():
    print("=== Proofreader: Native (Edge-and-Node) or Pipeline (LangGraph) Mode ===")
    file_path = input("Enter the path to the document file: ").strip()
    style = input("Enter style profile (us/uk) [default: us]: ").strip().lower() or 'us'
    mode = input("Select execution mode - 'native' (edge-and-node) or 'pipeline' (LangGraph) [default: pipeline]: ").strip().lower() or 'pipeline'
    assert mode in ['native', 'pipeline'], "Mode must be either 'native' or 'pipeline'"
    if mode == 'native':
        run_native(file_path, style)
    else:
        run_pipeline(file_path, style)

if __name__ == "__main__":
    main()
