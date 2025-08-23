import os, time, uuid, logging
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from src.document_processor import process_document
from src.vector_client import VectorClient
from src.model_loader import load_models
from src.neural_edit_agent import NeuralEditAgent, build_agent
from src.change_tracker import ValidatorAgent, ChangeTracker, generate_html_redline
from src.utils import save_json, append_jsonl

logger = logging.getLogger("langgraph_pipeline")
logger.setLevel(logging.INFO)

# Define the State schema
class ProofreadingState(TypedDict):
    job_id: str
    document_path: str
    style_profile: str
    doc_meta: Dict[str, Any]
    chunks: List[Dict]
    current_chunk_index: int
    retrieved_rules: List[Dict]
    instruction_block: str
    post_rules: Dict[str, str]
    protected_mask: List[Dict]
    original_text: str
    edited_text: str
    neural_meta: Dict[str, Any]
    validator_report: Dict[str, Any]
    changes: List[Dict]
    human_queue: List[Dict]
    logs: Annotated[List[str], operator.add]
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
    logger.info("Processing document: %s", state["document_path"])
    doc = process_document(state["document_path"])
    state["chunks"] = doc.get("chunks", [])
    state["current_chunk_index"] = 0
    state["logs"].append(f"Processed document into {len(state['chunks'])} chunks")
    return state

def retriever_node(state: ProofreadingState) -> ProofreadingState:
    chunk_index = state["current_chunk_index"]
    chunk = state["chunks"][chunk_index]
    rules = vector_client.retrieve(chunk["text"], style=state["style_profile"], k=12)
    state["retrieved_rules"] = rules
    state["logs"].append(f"Retrieved {len(rules)} rules for chunk {chunk_index}")
    return state

def style_knowledge_node(state: ProofreadingState) -> ProofreadingState:
    rules = state["retrieved_rules"]
    style = state["style_profile"]
    
    instruction_lines = []
    if style == "uk":
        instruction_lines.append("Apply UK English: -ise endings, single quotes.")
    elif style == "us":
        instruction_lines.append("Apply US English: -ize endings, double quotes.")
    
    for r in sorted(rules, key=lambda rr: -rr.get("priority", 0))[:4]:
        instruction_lines.append(f"- {r.get('content')}")
    
    post_rules = {}
    for r in rules:
        if r.get("mapping"):
            post_rules.update(r.get("mapping"))
    
    chunk = state["chunks"][state["current_chunk_index"]]
    protected_mask = chunk.get("protected_spans", [])
    
    state["instruction_block"] = "\n".join(instruction_lines)
    state["post_rules"] = post_rules
    state["protected_mask"] = protected_mask
    state["logs"].append("Processed style instructions and post-rules")
    
    return state

def neural_edit_node(state: ProofreadingState) -> ProofreadingState:
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
    report = validator_agent.validate(
        state["original_text"],
        state["edited_text"],
        {
            "gen_meta": state["neural_meta"],
            "rules": [r.get("rule_id") for r in state["retrieved_rules"]]
        }
    )
    
    state["validator_report"] = report
    state["logs"].append(f"Validated chunk {state['current_chunk_index']} (score: {report.get('validator_score', 0):.3f})")
    
    return state

def changelog_node(state: ProofreadingState) -> ProofreadingState:
    change_tracker.add_change(
        state["original_text"],
        state["edited_text"],
        state["current_chunk_index"],
        state["validator_report"]
    )
    
    html = generate_html_redline(
        state["original_text"],
        state["edited_text"],
        state["current_chunk_index"]
    )
    
    if "artifacts" not in state:
        state["artifacts"] = {}
    
    chunk_id = f"chunk_{state['current_chunk_index']}"
    state["artifacts"][f"{chunk_id}_redline"] = html
    
    state["logs"].append(f"Recorded changes for chunk {state['current_chunk_index']}")
    
    return state

def human_review_node(state: ProofreadingState) -> ProofreadingState:
    review_item = {
        "chunk_index": state["current_chunk_index"],
        "original_text": state["original_text"],
        "edited_text": state["edited_text"],
        "validator_report": state["validator_report"],
        "timestamp": time.time()
    }
    
    state["human_queue"].append(review_item)
    state["logs"].append(f"Added chunk {state['current_chunk_index']} to human review queue")
    
    return state

def progress_node(state: ProofreadingState) -> ProofreadingState:
    if state["current_chunk_index"] < len(state["chunks"]) - 1:
        state["current_chunk_index"] += 1
        state["logs"].append(f"Moving to chunk {state['current_chunk_index']}")
        return state
    else:
        state["logs"].append("Finished processing all chunks")
        aggregate = change_tracker.get_aggregate()
        state["artifacts"]["changelog"] = aggregate
        return state

def should_review(state: ProofreadingState) -> str:
    report = state["validator_report"]
    score = report.get("validator_score", 0)
    has_risky_changes = any(report.get("risk_flags", []))
    
    if score < 0.7 or has_risky_changes:
        return "human_review"
    return "approve"

def should_continue(state: ProofreadingState) -> str:
    if state["current_chunk_index"] < len(state["chunks"]) - 1:
        return "continue"
    return "end"

# Build the graph
def create_proofreading_graph() -> CompiledStateGraph:
    builder = StateGraph(ProofreadingState)
    
    builder.add_node("document_processor", document_processor_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("style_knowledge", style_knowledge_node)
    builder.add_node("neural_edit", neural_edit_node)
    builder.add_node("validator", validator_node)
    builder.add_node("changelog", changelog_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("progress", progress_node)
    
    builder.set_entry_point("document_processor")
    
    builder.add_edge("document_processor", "retriever")
    builder.add_edge("retriever", "style_knowledge")
    builder.add_edge("style_knowledge", "neural_edit")
    builder.add_edge("neural_edit", "validator")
    
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
    
    builder.add_conditional_edges(
        "progress",
        should_continue,
        {
            "continue": "retriever",
            "end": END
        }
    )
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# Create the graph instance
proofreading_graph = create_proofreading_graph()

# Main function to run the graph
async def process_document_with_graph(document_path: str, style_profile: str = "us") -> ProofreadingState:
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
    
    final_state = await proofreading_graph.ainvoke(initial_state)
    
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
    
    result = asyncio.run(process_document_with_graph(args.file, args.style))
    
    print(f"Job completed: {result['job_id']}")
    print(f"Processed {len(result['chunks'])} chunks")
    print(f"Changes made: {len(result.get('changes', []))}")
    print(f"Chunks needing review: {len(result.get('human_queue', []))}")
    
    save_json(f"artifacts/{result['job_id']}_final_state.json", result)