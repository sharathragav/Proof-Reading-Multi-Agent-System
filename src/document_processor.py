from typing import Dict, Any
import os
import logging
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
logger = logging.getLogger("document_processor")
logger.setLevel(logging.INFO)

def process_document(file_path: str) -> Dict[str, Any]:
    conv = DocumentConverter()
    result = conv.convert(file_path)
    chunker = HybridChunker()
    raw_chunks = list(chunker.chunk(result.document))
    chunks = []
    for i, raw in enumerate(raw_chunks):
        contextual = chunker.contextualize(raw)
        chunks.append({
                "chunk_id": f"{os.path.basename(file_path)}_chunk_{i}",
                "text": contextual.text,
                "block_refs": getattr(contextual, "block_refs", []),
                "start_offset": getattr(raw, "start", 0),
                "end_offset": getattr(raw, "end", len(contextual.text)),
                "protected_spans": getattr(raw, "protected_spans", []),
                "metadata": {}
            })
        blocks = [{"block_id": i, "type": getattr(b,"type","p"), "text": getattr(b,"text", "")} for i, b in enumerate(result.document.blocks)]
        return {"blocks": blocks, "chunks": chunks}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    return {"blocks": [{"block_id": "b0","type":"doc","text": txt}], "chunks": [{
            "chunk_id": f"{os.path.basename(file_path)}_chunk_0",
            "text": txt,
            "block_refs": [],
            "start_offset": 0,
            "end_offset": len(txt),
            "protected_spans": [],
            "metadata": {}
        }]}