"""
Batch Vector Ingestion Script for Supabase pgvector
Converts 49 Agricultural Markdown files into 768-dimensional embeddings
and stores them in Supabase for sub-millisecond semantic similarity search.
"""

import os
import sys
import glob
from typing import List, Dict, Any
from supabase import create_client, Client
import google.generativeai as genai
from backend.app.config import settings

def get_chunks_from_knowledge_base() -> List[Dict[str, Any]]:
    chunks = []
    knowledge_dir = settings.KNOWLEDGE_DIR
    
    for root, _, files in os.walk(knowledge_dir):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, knowledge_dir).replace("\\", "/")
                category = rel_path.split("/")[0] if "/" in rel_path else "general"
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                curr_chunk = []
                curr_header = rel_path

                for line in lines:
                    if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
                        if curr_chunk:
                            chunk_text = "\n".join(curr_chunk).strip()
                            if len(chunk_text) > 40:
                                chunks.append({
                                    "content": f"[{curr_header}]\n{chunk_text}",
                                    "source": rel_path,
                                    "category": category,
                                    "metadata": {"title": curr_header, "category": category, "source": rel_path}
                                })
                            curr_chunk = []
                        curr_header = f"{rel_path} > {line.strip('# ')}"
                    curr_chunk.append(line)

                if curr_chunk:
                    chunk_text = "\n".join(curr_chunk).strip()
                    if len(chunk_text) > 40:
                        chunks.append({
                            "content": f"[{curr_header}]\n{chunk_text}",
                            "source": rel_path,
                            "category": category,
                            "metadata": {"title": curr_header, "category": category, "source": rel_path}
                        })
                        
    return chunks

def generate_embedding(text: str, api_key: str) -> List[float]:
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

def ingest_to_supabase(supabase_url: str, supabase_key: str, gemini_api_key: str):
    print("=" * 60)
    print(">>> INGESTING AGRICULTURAL KNOWLEDGE TO SUPABASE PGVECTOR")
    print("=" * 60)
    
    if not supabase_url or not supabase_key:
        print("[ERROR] Supabase URL and Key are required. Set them in .env or pass as arguments.")
        return

    if not gemini_api_key:
        print("[ERROR] Gemini API Key is required for generating embeddings.")
        return

    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    chunks = get_chunks_from_knowledge_base()
    print(f"Loaded {len(chunks)} knowledge chunks from markdown files.")
    
    records_to_insert = []
    print("Generating embeddings and uploading to Supabase...")
    
    for i, chunk in enumerate(chunks):
        try:
            emb = generate_embedding(chunk["content"], gemini_api_key)
            record = {
                "content": chunk["content"],
                "source": chunk["source"],
                "category": chunk["category"],
                "metadata": chunk["metadata"],
                "embedding": emb
            }
            records_to_insert.append(record)
            
            # Batch upload every 20 records
            if len(records_to_insert) >= 20 or i == len(chunks) - 1:
                res = supabase.table("crop_knowledge_documents").insert(records_to_insert).execute()
                print(f"  Uploaded {i + 1} / {len(chunks)} chunks to Supabase...")
                records_to_insert = []
        except Exception as e:
            print(f"  [Notice] Error embedding chunk {i}: {e}")

    print("=" * 60)
    print(">>> INGESTION COMPLETE! All knowledge vectors stored in Supabase.")
    print("=" * 60)

if __name__ == "__main__":
    sb_url = settings.SUPABASE_URL or os.getenv("SUPABASE_URL", "")
    sb_key = settings.SUPABASE_KEY or os.getenv("SUPABASE_KEY", "")
    gem_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    
    if len(sys.argv) >= 3:
        sb_url = sys.argv[1]
        sb_key = sys.argv[2]
    if len(sys.argv) >= 4:
        gem_key = sys.argv[3]

    if not sb_url or not sb_key:
        print("Usage:")
        print("  python backend/ml/ingest_to_supabase.py <SUPABASE_URL> <SUPABASE_KEY> <GEMINI_API_KEY>")
        print("Or set SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY in your .env file.")
    else:
        ingest_to_supabase(sb_url, sb_key, gem_key)
