import difflib, uuid, time, json, os
from typing import Dict, Any
from spellchecker import SpellChecker
import language_tool_python
import spacy
from diff_match_patch import diff_match_patch
from src.utils import save_json, append_jsonl

nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool("en-US")
spell = SpellChecker()

class ValidatorAgent:
    def __init__(self, style="us"):
        self.style = style

    def validate(self, original: str, edited: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        report = {"validator_id": str(uuid.uuid4()), "timestamp": time.time(), "ops": [], "validator_score":0.0, "risk_flags": [], "applied_rules": metadata.get("rules", [])}
        s = difflib.SequenceMatcher(a=original.split(), b=edited.split())
        for tag,i1,i2,j1,j2 in s.get_opcodes():
            if tag == "equal":
                continue
            orig_piece = " ".join(original.split()[i1:i2]) if i2>i1 else ""
            new_piece = " ".join(edited.split()[j1:j2]) if j2>j1 else ""
            classification = "word_choice"
            if any(ch.isdigit() for ch in orig_piece+new_piece):
                classification = "numeric_change"
            elif orig_piece.strip(".,;:!?") != orig_piece or new_piece.strip(".,;:!?") != new_piece:
                classification = "punctuation"
            elif orig_piece.lower() not in spell and new_piece.lower() in spell:
                classification = "spelling"
            else:
                o_doc = nlp(orig_piece[:1000])
                n_doc = nlp(new_piece[:1000])
                if o_doc and n_doc and o_doc[0].pos_ != n_doc[0].pos_:
                    classification = "grammar"
            report["ops"].append({"op":tag,"orig":orig_piece,"new":new_piece,"classification":classification})
            if classification in ("numeric_change","protected_change"):
                report["risk_flags"].append(classification)

        orig_matches = tool.check(original)
        edited_matches = tool.check(edited)
        tool_score = 1.0 - (len(edited_matches) / (len(orig_matches)+1e-6))
        model_conf = metadata.get("gen_meta", {}).get("beam_score", 0.8)
        rag_strength = 1.0 if metadata.get("rules") else 0.2
        post_match = 1.0 if any(op["classification"]=="spelling" for op in report["ops"]) and metadata.get("rules") else 0.0
        score = 0.45*model_conf + 0.2*rag_strength + 0.2*max(0, tool_score) + 0.1*post_match
        if "protected_change" in report["risk_flags"]:
            score *= 0.1
        report["validator_score"] = max(0.0, min(1.0, score))
        return report

class ChangeTracker:
    def __init__(self, outdir="artifacts"):
        self.changes = []
        self.stats = {"total_chunks":0, "by_type":{}}
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def add_change(self, original: str, proofread: str, chunk_index: int, validator_report: Dict[str, Any]):
        entry = {
            "edit_id": str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "orig_text": original,
            "new_text": proofread,
            "ops": validator_report.get("ops", []),
            "validator_score": validator_report.get("validator_score", 0.0),
            "applied_rules": validator_report.get("applied_rules", []),
            "timestamp": validator_report.get("timestamp"),
            "review_status": "auto_applied" if validator_report.get("validator_score",0)>=0.85 and not validator_report.get("risk_flags") else "pending",
            "risk_flags": validator_report.get("risk_flags", []),
            "validator_id": validator_report.get("validator_id")
        }
        self.changes.append(entry)
        self.stats["total_chunks"] += 1
        for op in entry["ops"]:
            t = op.get("classification","unknown")
            self.stats["by_type"][t] = self.stats["by_type"].get(t,0) + 1
        out_path = os.path.join(self.outdir, f"changelog_{int(time.time())}.jsonl")
        append_jsonl(out_path, entry)

    def get_aggregate(self):
        return {"stats": self.stats, "changes_count": len(self.changes), "changes": self.changes}

def generate_html_redline(original: str, edited: str, chunk_id: int) -> str:
    dmp = diff_match_patch()
    diffs = dmp.diff_main(original, edited)
    dmp.diff_cleanupSemantic(diffs)
    html = []
    for op, data in diffs:
        if op == dmp.DIFF_INSERT:
            html.append(f"<ins data-chunk='{chunk_id}'>{data}</ins>")
        elif op == dmp.DIFF_DELETE:
            html.append(f"<del data-chunk='{chunk_id}'>{data}</del>")
        else:
            html.append(data)
    return "<div class='chunk' id='chunk-{}'>{}</div>".format(chunk_id, "".join(html))