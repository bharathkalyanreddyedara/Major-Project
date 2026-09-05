import os
import glob
from typing import List, Dict, Any
from backend.app.config import settings

class RAGAssistantService:
    def __init__(self):
        self.documents = []
        self.supabase_client = None
        self.init_supabase()
        self.load_knowledge_base()

    def init_supabase(self):
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                from supabase import create_client
                self.supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                print("[RAGAssistantService] Connected to Supabase pgvector database!")
            except Exception as e:
                print(f"[RAGAssistantService] Supabase connection notice: {e}")

    def load_knowledge_base(self):
        self.documents = []
        if not os.path.exists(settings.KNOWLEDGE_DIR):
            os.makedirs(settings.KNOWLEDGE_DIR, exist_ok=True)

        # Recursively search for all .md and .txt files in knowledge directory
        knowledge_files = []
        for root, _, files in os.walk(settings.KNOWLEDGE_DIR):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    knowledge_files.append(os.path.join(root, file))

        for file_path in knowledge_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    rel_source = os.path.relpath(file_path, settings.KNOWLEDGE_DIR).replace("\\", "/")
                    
                    # Split by markdown headers (# or ##)
                    chunks = []
                    lines = content.split("\n")
                    curr_chunk = []
                    curr_header = rel_source

                    for line in lines:
                        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
                            if curr_chunk:
                                chunk_text = "\n".join(curr_chunk).strip()
                                if len(chunk_text) > 40:
                                    chunks.append(f"[{curr_header}]\n{chunk_text}")
                                curr_chunk = []
                            curr_header = f"{rel_source} > {line.strip('# ')}"
                        curr_chunk.append(line)

                    if curr_chunk:
                        chunk_text = "\n".join(curr_chunk).strip()
                        if len(chunk_text) > 40:
                            chunks.append(f"[{curr_header}]\n{chunk_text}")

                    for chunk in chunks:
                        self.documents.append({
                            "source": rel_source,
                            "text": chunk
                        })
            except Exception as e:
                print(f"[RAGAssistantService] Error reading knowledge file {file_path}: {e}")
        
        print(f"[RAGAssistantService] Indexed {len(self.documents)} dynamic knowledge chunks from {len(knowledge_files)} Markdown/text files.")

    def retrieve_relevant_docs(self, query: str, context_tags: List[str] = None, top_k: int = 4) -> List[Dict[str, Any]]:
        # 1. First Attempt: Supabase pgvector Similarity Search (if configured)
        if self.supabase_client and settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                emb_res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                query_vec = emb_res["embedding"]
                rpc_res = self.supabase_client.rpc(
                    "match_crop_documents",
                    {
                        "query_embedding": query_vec,
                        "match_threshold": 0.25,
                        "match_count": top_k
                    }
                ).execute()

                if rpc_res.data and len(rpc_res.data) > 0:
                    return [{"source": r["source"], "text": r["content"], "similarity": r.get("similarity")} for r in rpc_res.data]
            except Exception as e:
                print(f"[RAGAssistantService] Supabase vector query notice: {e}")

        # 2. In-Memory Grounded Search Fallback
        query_words = [w.lower() for w in query.replace("?", "").replace(",", "").split() if len(w) > 2]
        if context_tags:
            for tag in context_tags:
                if tag:
                    query_words.extend([w.lower() for w in tag.split() if len(w) > 2])

        scored_docs = []
        for doc in self.documents:
            text_lower = doc["text"].lower()
            score = 0
            for word in query_words:
                if word in text_lower:
                    score += text_lower.count(word)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]] if scored_docs else self.documents[:2]

    def answer_query(
        self,
        query: str,
        crop_context: str = None,
        soil_context: str = None,
        growth_stage_context: str = None,
        weather_context: str = None,
        history: List[Any] = None
    ) -> Dict[str, Any]:
        
        # 1. Retrieve grounded docs
        relevant_docs = self.retrieve_relevant_docs(query)
        grounded_context_str = "\n\n".join([f"Source [{d['source']}]:\n{d['text']}" for d in relevant_docs])
        sources = list(set([d["source"] for d in relevant_docs]))

        # 2. Build system prompt injecting full farmer context
        system_context = (
            "You are an expert AI Agricultural Assistant & Agronomist supporting a farmer.\n"
            "Provide clear, actionable, and scientific farming guidance. Always ground your answers in the agricultural knowledge provided.\n\n"
            f"FARMER CONTEXT:\n"
            f"- Active Crop: {crop_context or 'Not selected / General'}\n"
            f"- Soil Type & Properties: {soil_context or 'Unspecified'}\n"
            f"- Current Growth Stage: {growth_stage_context or 'Planning / Sowing'}\n"
            f"- Local Weather: {weather_context or 'Normal seasonal conditions'}\n\n"
            f"GROUNDED AGRICULTURAL KNOWLEDGE:\n"
            f"{grounded_context_str}\n"
        )

        # 3. Call Gemini API if key is present
        api_key = settings.GEMINI_API_KEY
        answer = ""
        suggested_actions = [
            "Check current soil moisture before watering",
            "Monitor for early pest infestation on leaf undersides",
            "Refer to the stage-specific fertilizer schedule"
        ]

        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                full_prompt = f"{system_context}\n\nFarmer Query: {query}\n\nExpert Agronomist Response:"
                response = model.generate_content(full_prompt)
                answer = response.text
            except Exception as e:
                print(f"Gemini API invocation error: {e}")

        if not answer:
            # High-quality contextual extraction directly from grounded Markdown chunks
            chunk_excerpts = []
            for d in relevant_docs[:3]:
                clean_text = d["text"].strip()
                # Remove header marker for cleaner reading if present
                clean_text = clean_text.replace("# ", "").replace("## ", "• ").replace("### ", "  - ")
                chunk_excerpts.append(f"📖 **From `{d['source']}`:**\n{clean_text}")

            extracted_knowledge = "\n\n---\n\n".join(chunk_excerpts)

            answer = (
                f"### 🌾 Agronomic Advisory for **{crop_context or 'Crop'}** on **{soil_context or 'Soil'}**\n\n"
                f"**Current Status:** {growth_stage_context or 'Vegetative Phase'} | **Weather:** {weather_context or 'Clear'}\n\n"
                f"{extracted_knowledge}\n\n"
                f"**Actionable Next Steps:**\n"
                f"- Ensure proper drainage channels if heavy precipitation is forecast.\n"
                f"- Verify soil moisture prior to top-dressing or fertigation.\n"
                f"- Adhere to stage-specific NPK split dosages and scout for ETL pest thresholds."
            )

        return {
            "answer": answer,
            "grounded_sources": sources,
            "suggested_actions": suggested_actions
        }

rag_assistant_service = RAGAssistantService()
