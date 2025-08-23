import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("vector_client")
logger.setLevel(logging.INFO)

# ChromaDB imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configuration
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_IMPL = os.environ.get("CHROMA_IMPL", "duckdb+parquet")

class VectorClient:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        # Initialize ChromaDB client
        settings = Settings(
            chroma_db_impl=CHROMA_IMPL,
            persist_directory=CHROMA_PERSIST_DIR
        )
        self.client = chromadb.Client(settings=settings)
        self.collection = self.client.get_or_create_collection(name="rules")
        
        logger.info(f"ChromaDB client initialized with persistence directory: {CHROMA_PERSIST_DIR}")

    def ensure_schema(self):
        # In ChromaDB, schema is automatically handled
        pass

    def index_rule(self, rule_doc: Dict[str, Any], embedding: Optional[List[float]] = None):
        # Prepare metadata
        metadata = {
            "rule_id": rule_doc["rule_id"],
            "style_tags": rule_doc.get("style_tags", []),
            "priority": float(rule_doc.get("priority", 50)),
            "mapping": json.dumps(rule_doc.get("mapping", {})),
            "examples": json.dumps(rule_doc.get("examples", []))
        }
        
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embedding_model.encode(rule_doc["content"]).tolist()
        
        # Add to collection
        self.collection.add(
            ids=[rule_doc["rule_id"]],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[rule_doc["content"]]
        )
        
        # Persist data
        self.client.persist()
        logger.info(f"Indexed rule: {rule_doc['rule_id']}")

    def retrieve(self, text: str, style: Optional[str] = None, k: int = 12) -> List[Dict[str, Any]]:
        try:
            # Compute query embedding
            query_embedding = self.embedding_model.encode(text).tolist()
            
            # Retrieve results
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 3,  # Get more results for filtering
                include=["metadatas", "documents", "distances"]
            )
            
            # Process results
            rules = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    rule_id = results["ids"][0][i]
                    content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else None
                    
                    # Filter by style if specified
                    if style and style not in metadata.get("style_tags", []):
                        continue
                    
                    # Parse metadata
                    rule = {
                        "rule_id": rule_id,
                        "content": content,
                        "priority": metadata.get("priority", 50),
                        "mapping": json.loads(metadata.get("mapping", "{}")),
                        "examples": json.loads(metadata.get("examples", "[]")),
                        "distance": distance
                    }
                    
                    rules.append(rule)
                    
                    if len(rules) >= k:
                        break
            
            return rules
        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            return []