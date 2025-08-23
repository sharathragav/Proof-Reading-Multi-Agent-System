import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vector_client import VectorClient

def main(rules_path):
    vc = VectorClient()
    vc.ensure_schema()
    
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    
    for rule in rules:
        vc.index_rule(rule)
    
    print(f"Indexed {len(rules)} rules into ChromaDB at {os.environ.get('CHROMA_PERSIST_DIR', './chroma_db')}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python index_rules_chroma.py path/to/rules.json")
        sys.exit(1)
    
    main(sys.argv[1])