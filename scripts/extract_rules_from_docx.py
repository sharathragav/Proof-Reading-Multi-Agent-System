from docx import Document
import json
import re

def extract_rules_from_docx(docx_path, output_json):
    doc = Document(docx_path)
    text = '\n'.join(para.text for para in doc.paragraphs)
    # Example: Each rule starts with 'Rule ID:' and is separated by blank lines
    rule_blocks = re.split(r'\n\s*Rule ID:', text)
    rules = []
    for block in rule_blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        rule = {}
        rule['rule_id'] = lines[0].strip() if lines else ''
        for line in lines[1:]:
            if line.startswith('Content:'):
                rule['content'] = line.replace('Content:', '').strip()
            elif line.startswith('Priority:'):
                rule['priority'] = float(line.replace('Priority:', '').strip())
            elif line.startswith('Mapping:'):
                try:
                    rule['mapping'] = json.loads(line.replace('Mapping:', '').strip())
                except Exception:
                    rule['mapping'] = {}
            elif line.startswith('Examples:'):
                try:
                    rule['examples'] = json.loads(line.replace('Examples:', '').strip())
                except Exception:
                    rule['examples'] = []
        rules.append(rule)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

# Usage:
# extract_rules_from_docx('rules.docx', 'rules.json')
