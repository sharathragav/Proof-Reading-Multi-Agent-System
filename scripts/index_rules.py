import sys, os, json
from sentence_transformers import SentenceTransformer
from src.vector_client import VectorClient

def main(rules_path):
    vc = VectorClient()
    vc.ensure_schema()
    emb_model = SentenceTransformer(os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"))
    with open(rules_path, "r", encoding="utf-8") as fh:
        rules = json.load(fh)
    for r in rules:
        vec = emb_model.encode(r.get("content","")).tolist()
        vc.index_rule(r, embedding=vec)
    print("Indexed", len(rules), "rules.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_rules.py path/to/rules.json")
        sys.exit(1)
    main(sys.argv[1])