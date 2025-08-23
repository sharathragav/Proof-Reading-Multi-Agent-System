# src/langgraph_native.py
"""
LangGraph-native implementation of the Enterprise Proofreader.
This version uses the official LangGraph API with proper state management.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
import operator
import uuid
import logging
from datetime import datetime

from src.document_processor import process_document
from src.vector_client import VectorClient
from src.model_loader import load_models
from src.neural_edit_agent import NeuralEditAgent, build_agent
from src.change_tracker import ValidatorAgent, ChangeTracker, generate_html_redline
from src.utils import save_json, append_jsonl

logger = logging.getLogger("langgraph_native")
logger.setLevel(logging.INFO)

# Define the State schema
class ProofreadingState(TypedDict):
    # Document information
    job_id: str
    document_path: str
    style_profile: str
    doc_meta: Dict[str, Any]
    
    # Processing state
    chunks: List[Dict]
    current_chunk_index: int
    
    # RAG context
    retrieved_rules: List[Dict]
    instruction_block: str
    post_rules: Dict[str, str]
    protected_mask: List[Dict]
    
    # Editing results
    original_text: str
    edited_text: str
    neural_meta: Dict[str, Any]
    
    # Validation results
    validator_report: Dict[str, Any]
    
    # Tracking
    changes: List[Dict]
    human_queue: List[Dict]
    
    # Logs
    logs: Annotated[List[str], operator.add]
    
    # Final output
    artifacts: Dict[str, Any]

# Initialize shared components
vector_client = VectorClient()
validator_agent = ValidatorAgent()
change_tracker = ChangeTracker()

# Load models once (shared across nodes)
small_tok, small_model, small_dev, big_tok, big_model, big_dev = load_models()
neural_agent = NeuralEditAgent(small_tok, small_model, small_dev, big_tok, big_model, big_dev)

# Define node functions
def document_processor_node(state: ProofreadingState) -> ProofreadingState:
    """Process document using docling or fallback"""
    logger.info("Processing document: %s", state["document_path"])
    doc = process_document(state["document_path"])
    state["chunks"] = doc.get("chunks", [])
    state["current_chunk_index"] = 0
    state["logs"].append(f"Processed document into {len(state['chunks'])} chunks")
    return state

def retriever_node(state: ProofreadingState) -> ProofreadingState:
    """Retrieve relevant style rules for current chunk"""
    chunk_index = state["current_chunk_index"]
    chunk = state["chunks"][chunk_index]
    rules = vector_client.retrieve(
        chunk["text"], 
        style=state["style_profile"], 
        k=12
    )
    state["retrieved_rules"] = rules
    state["logs"].append(f"Retrieved {len(rules)} rules for chunk {chunk_index}")
    return state

def style_knowledge_node(state: ProofreadingState) -> ProofreadingState:
    """Process rules into instructions and post-processing rules"""
    rules = state["retrieved_rules"]
    style = state["style_profile"]
    
    # Build instruction block
    instruction_lines = []
    if style == "uk":
        instruction_lines.append("Apply UK English: -ise endings, single quotes.")
    elif style == "us":
        instruction_lines.append("Apply US English: -ize endings, double quotes.")
    
    for r in sorted(rules, key=lambda rr: -rr.get("priority", 0))[:4]:
        instruction_lines.append(f"- {r.get('content')}")
    
    # Build post-processing rules
    post_rules = {}
    for r in rules:
        if r.get("mapping"):
            post_rules.update(r.get("mapping"))
    
    # Get protected spans from current chunk
    chunk = state["chunks"][state["current_chunk_index"]]
    protected_mask = chunk.get("protected_spans", [])
    
    state["instruction_block"] = "\n".join(instruction_lines)
    state["post_rules"] = post_rules
    state["protected_mask"] = protected_mask
    state["logs"].append("Processed style instructions and post-rules")
    
    return state

def neural_edit_node(state: ProofreadingState) -> ProofreadingState:
    """Apply neural editing to current chunk"""
    chunk_index = state["current_chunk_index"]
    chunk = state["chunks"][chunk_index]
    
    edited, gen_meta = neural_agent.edit_chunk(
        chunk["text"],
        state["instruction_block"],
        exemplars=state["retrieved_rules"][:3],
        post_rules=state["post_rules"],
        protected_mask=state["protected_mask"]
    )
    
    state["original_text"] = chunk["text"]
    state["edited_text"] = edited
    state["neural_meta"] = gen_meta
    state["logs"].append(f"Edited chunk {chunk_index} (path: {gen_meta.get('path')})")
    
    return state

def validator_node(state: ProofreadingState) -> ProofreadingState:
    """Validate edits and compute confidence score"""
    report = validator_agent.validate(
        state["original_text"],
        state["edited_text"],
        {
            "gen_meta": state["neural_meta"],
            "rules": [r.get("rule_id") for r in state["retrieved_rules"]]
        }
    )
    
    state["validator_report"] = report
    state["logs"].append(
        f"Validated chunk {state['current_chunk_index']} "
        f"(score: {report.get('validator_score', 0):.3f})"
    )
    
    return state

def changelog_node(state: ProofreadingState) -> ProofreadingState:
    """Record changes and generate artifacts"""
    change_tracker.add_change(
        state["original_text"],
        state["edited_text"],
        state["current_chunk_index"],
        state["validator_report"]
    )
    
    # Generate HTML redline
    html = generate_html_redline(
        state["original_text"],
        state["edited_text"],
        state["current_chunk_index"]
    )
    
    # Store artifacts in state
    if "artifacts" not in state:
        state["artifacts"] = {}
    
    chunk_id = f"chunk_{state['current_chunk_index']}"
    state["artifacts"][f"{chunk_id}_redline"] = html
    
    state["logs"].append(f"Recorded changes for chunk {state['current_chunk_index']}")
    
    return state

def human_review_node(state: ProofreadingState) -> ProofreadingState:
    """Handle human review for low-confidence changes"""
    review_item = {
        "chunk_index": state["current_chunk_index"],
        "original_text": state["original_text"],
        "edited_text": state["edited_text"],
        "validator_report": state["validator_report"],
        "timestamp": datetime.now().isoformat()
    }
    
    state["human_queue"].append(review_item)
    state["logs"].append(
        f"Added chunk {state['current_chunk_index']} to human review queue"
    )
    
    return state

def progress_node(state: ProofreadingState) -> ProofreadingState:
    """Move to next chunk or finish processing"""
    if state["current_chunk_index"] < len(state["chunks"]) - 1:
        state["current_chunk_index"] += 1
        state["logs"].append(f"Moving to chunk {state['current_chunk_index']}")
        return state
    else:
        state["logs"].append("Finished processing all chunks")
        # Generate final artifacts
        aggregate = change_tracker.get_aggregate()
        state["artifacts"]["changelog"] = aggregate
        return state

def should_review(state: ProofreadingState) -> str:
    """Conditional edge: determine if changes need human review"""
    report = state["validator_report"]
    score = report.get("validator_score", 0)
    has_risky_changes = any(report.get("risk_flags", []))
    
    if score < 0.7 or has_risky_changes:
        return "human_review"
    return "approve"

def should_continue(state: ProofreadingState) -> str:
    """Conditional edge: determine if we should process next chunk"""
    if state["current_chunk_index"] < len(state["chunks"]) - 1:
        return "continue"
    return "end"

# Build the graph
def create_proofreading_graph() -> CompiledStateGraph:
    """Create and compile the proofreading state graph"""
    builder = StateGraph(ProofreadingState)
    
    # Add nodes
    builder.add_node("document_processor", document_processor_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("style_knowledge", style_knowledge_node)
    builder.add_node("neural_edit", neural_edit_node)
    builder.add_node("validator", validator_node)
    builder.add_node("changelog", changelog_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("progress", progress_node)
    
    # Set entry point
    builder.set_entry_point("document_processor")
    
    # Add edges
    builder.add_edge("document_processor", "retriever")
    builder.add_edge("retriever", "style_knowledge")
    builder.add_edge("style_knowledge", "neural_edit")
    builder.add_edge("neural_edit", "validator")
    
    # Conditional edge after validator
    builder.add_conditional_edges(
        "validator",
        should_review,
        {
            "human_review": "human_review",
            "approve": "changelog"
        }
    )
    
    builder.add_edge("human_review", "changelog")
    builder.add_edge("changelog", "progress")
    
    # Conditional edge after progress
    builder.add_conditional_edges(
        "progress",
        should_continue,
        {
            "continue": "retriever",
            "end": END
        }
    )
    
    # Compile with memory saver for checkpointing
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# Create the graph instance
proofreading_graph = create_proofreading_graph()

# Main function to run the graph
async def process_document_with_graph(document_path: str, style_profile: str = "us") -> ProofreadingState:
    """Process a document using the LangGraph proofreading pipeline"""
    initial_state = {
        "job_id": str(uuid.uuid4()),
        "document_path": document_path,
        "style_profile": style_profile,
        "doc_meta": {},
        "chunks": [],
        "current_chunk_index": 0,
        "retrieved_rules": [],
        "instruction_block": "",
        "post_rules": {},
        "protected_mask": [],
        "original_text": "",
        "edited_text": "",
        "neural_meta": {},
        "validator_report": {},
        "changes": [],
        "human_queue": [],
        "logs": [],
        "artifacts": {}
    }
    
    # Run the graph
    final_state = await proofreading_graph.ainvoke(initial_state)
    
    # Save final artifacts
    job_id = final_state["job_id"]
    save_json(f"artifacts/{job_id}_final_changelog.json", final_state["artifacts"].get("changelog", {}))
    
    for name, content in final_state["artifacts"].items():
        if name.endswith("_redline"):
            with open(f"artifacts/{job_id}_{name}.html", "w", encoding="utf-8") as f:
                f.write(content)
    
    return final_state

# CLI entry point
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Process a document with LangGraph proofreading")
    parser.add_argument("--file", required=True, help="Path to document file")
    parser.add_argument("--style", default="us", help="Style profile (us/uk)")
    
    args = parser.parse_args()
    
    # Run the graph
    result = asyncio.run(process_document_with_graph(args.file, args.style))
    
    # Print summary
    print(f"Job completed: {result['job_id']}")
    print(f"Processed {len(result['chunks'])} chunks")
    print(f"Changes made: {len(result.get('changes', []))}")
    print(f"Chunks needing review: {len(result.get('human_queue', []))}")
    
    # Save final state
    save_json(f"artifacts/{result['job_id']}_final_state.json", result)