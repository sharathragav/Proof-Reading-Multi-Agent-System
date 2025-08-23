import time, re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
from src.model_loader import load_models
import torch
try:
    import language_tool_python
    TOOL_AVAILABLE = True
    LANG_TOOL = language_tool_python.LanguageTool("en-US")
except Exception:
    TOOL_AVAILABLE = False

class NeuralEditAgent:
    def __init__(self, small_tokenizer, small_model, small_device, tokenizer, model, device):
        self.small_tok = small_tokenizer
        self.small_model = small_model
        self.small_dev = small_device
        self.big_tok = tokenizer
        self.big_model = model
        self.big_dev = device

    def _generate(self, tok, model, dev, prompt, max_in=512, max_out=512):
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in).to(dev)
        with torch.no_grad():
            outs = model.generate(**inputs, max_length=max_out, num_beams=4, do_sample=False, early_stopping=True, no_repeat_ngram_size=3)
        return tok.decode(outs[0], skip_special_tokens=True).strip()

    def _similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def _apply_post_rules(self, text, post_rules):
        for k,v in (post_rules or {}).items():
            text = re.sub(r'\b{}\b'.format(re.escape(k)), v, text, flags=re.IGNORECASE)
        return text

    def edit_chunk(self, chunk_text: str, instruction_block: str, exemplars: List[Dict]=None, post_rules: Dict=None, protected_mask: List[Dict]=None) -> Tuple[str, Dict]:
        exemplars = exemplars or []
        post_rules = post_rules or {}
        protected_mask = protected_mask or []

        # Build a detailed, context-rich prompt
        parts = []
        parts.append("You are a professional proofreader and editor. Your task is to edit the following document chunk according to the provided style guide and rules.")
        if instruction_block:
            parts.append("Style Guide and Rules:\n" + instruction_block)
        if exemplars:
            exs = []
            for e in exemplars[:3]:
                orig = e.get("orig") or e.get("original","")
                ed = e.get("edited") or e.get("edited_text","")
                exs.append(f"Original: {orig}\nEdited: {ed}")
            parts.append("Editing Examples (before and after):\n" + "\n\n".join(exs))
        parts.append("Additional Instructions:")
        parts.append("- Do not change anything in $...$ (LaTeX), DOIs, URLs, or code blocks.")
        parts.append("- Preserve the meaning and intent of the original text.")
        parts.append("- Apply all relevant rules and style requirements strictly.")
        parts.append("- If the chunk is already correct, make minimal or no changes.")
        parts.append("")
        parts.append("Document Chunk to Edit:\n" + chunk_text)
        parts.append("")
        parts.append("Output Requirements:")
        parts.append("- Output only the fully edited chunk text, nothing else.")
        parts.append("- Do not include explanations or formatting.")
        prompt = "\n\n".join(parts)

        t0 = time.time()

        try:
            inputs = self.small_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.small_dev)
            with torch.no_grad():
                outs = self.small_model.generate(**inputs, max_length=512, num_beams=4, do_sample=False)
            quick_out = self.small_tok.decode(outs[0], skip_special_tokens=True).strip()
            sim = self._similarity(chunk_text, quick_out)
        except Exception:
            quick_out = None
            sim = 0.0

        escalate = True
        if quick_out is not None:
            if sim > 0.92:
                escalate = False
            elif TOOL_AVAILABLE:
                orig_issues = len(LANG_TOOL.check(chunk_text))
                quick_issues = len(LANG_TOOL.check(quick_out))
                if quick_issues <= (orig_issues * 0.6):
                    escalate = False

        if not escalate:
            edited = quick_out
            gen_path = "small_model"
        else:
            inputs = self.big_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.big_dev)
            with torch.no_grad():
                outs = self.big_model.generate(**inputs, max_length=512, num_beams=4, do_sample=False)
            edited = self.big_tok.decode(outs[0], skip_special_tokens=True).strip()
            gen_path = "big_model"

        edited = self._apply_post_rules(edited, post_rules)

        if protected_mask:
            for span in protected_mask:
                s = span.get("start_rel"); e = span.get("end_rel")
                if s is None or e is None:
                    continue
                edited = edited[:s] + chunk_text[s:e] + edited[e:]

        gen_meta = {"path": gen_path, "time_s": time.time()-t0, "len_orig": len(chunk_text), "len_edited": len(edited)}
        return edited, gen_meta

def build_agent(small_name="google/flan-t5-small", big_name="coedit-large-local", use_8bit=False):
    small_tok, small_model, small_dev, big_tok, big_model, big_dev = load_models(small_name, big_name, use_8bit)
    return NeuralEditAgent(small_tok, small_model, small_dev, big_tok, big_model, big_dev)