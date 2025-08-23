import pdfplumber
import json
import re

def extract_rules_from_pdf(pdf_path, output_json):
    rules = []
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    # Example: Each rule starts with 'Rule ID:' and is separated by blank lines
    rule_blocks = re.split(r'\n\s*Rule ID:', text)
    for block in rule_blocks:
        if not block.strip():
            continue
        # Simple parsing logic (customize as needed)
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
# extract_rules_from_pdf('rules.pdf', 'rules.json')
