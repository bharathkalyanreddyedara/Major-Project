import os
import glob
import re
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
                    
                    chunks = []
                    lines = content.split("\n")
                    curr_chunk = []
                    curr_header = rel_source

                    for line in lines:
                        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
                            if curr_chunk:
                                chunk_text = "\n".join(curr_chunk).strip()
                                if len(chunk_text) > 30:
                                    chunks.append({
                                        "header": curr_header,
                                        "text": chunk_text
                                    })
                                curr_chunk = []
                            curr_header = f"{rel_source} > {line.strip('# ')}"
                        curr_chunk.append(line)

                    if curr_chunk:
                        chunk_text = "\n".join(curr_chunk).strip()
                        if len(chunk_text) > 30:
                            chunks.append({
                                "header": curr_header,
                                "text": chunk_text
                            })

                    for chunk in chunks:
                        self.documents.append({
                            "source": rel_source,
                            "header": chunk["header"],
                            "text": f"[{chunk['header']}]\n{chunk['text']}"
                        })
            except Exception as e:
                print(f"[RAGAssistantService] Error reading knowledge file {file_path}: {e}")
        
        print(f"[RAGAssistantService] Indexed {len(self.documents)} dynamic knowledge chunks from {len(knowledge_files)} Markdown/text files.")

    def is_conversational_query(self, query: str) -> tuple[bool, str]:
        """Detects greetings, pleasantries, and meta-chat queries to respond naturally without raw document dumps."""
        q_clean = re.sub(r"[^\w\s]", "", query.strip().lower())
        
        # 1. Direct Greetings
        greetings = ["hi", "hello", "hey", "heya", "namaste", "namaskar", "vanakkam", "pranam", "good morning", "good afternoon", "good evening", "good day"]
        if q_clean in greetings or any(q_clean.startswith(g + " ") for g in greetings):
            return True, (
                "Hello! 👋 I am your **AI Agronomist & Precision Farm Assistant**.\n\n"
                "I can help you with:\n"
                "- 🌾 **Crop Recommendations & Suitability Planning**\n"
                "- 🧪 **Soil Health, NPK Deficit Formulas & Micro-Nutrient Fixes**\n"
                "- 💧 **Drip Fertigation & Water-Soluble Fertilizer (WSF) Schedules**\n"
                "- 🐛 **Pest & Disease Diagnosis, ETL Thresholds & IPM Solutions**\n"
                "- 🌿 **Natural & Organic Farming (Jeevamrutham, Panchagavya, Bio-fertilizers)**\n"
                "- 🏛️ **Government Subsidies (PMKSY, SMAM, PM-Kisan, PMFBY)**\n\n"
                "How can I assist your farm or crops today?"
            )

        # 2. How are you
        if any(p in q_clean for p in ["how are you", "how r u", "how do you do", "how is it going"]):
            return True, "I am doing great and ready to assist with your crops and soil health! 🌾 What crop or farming challenge would you like help with today?"

        # 3. Who are you / Identity
        if any(p in q_clean for p in ["who are you", "what are you", "what can you do", "introduce yourself", "your name", "what is your role"]):
            return True, (
                "I am your dedicated **AI Agricultural Assistant**, powered by verified agronomic knowledge from "
                "ICAR, ICRISAT, and AgricultureGuruji. I provide scientifically grounded crop advisories, 50-kg commercial fertilizer "
                "bag calculations, proactive weather-risk mitigation, and stage-wise crop management guidance."
            )

        # 4. Gratitude
        if any(p in q_clean for p in ["thanks", "thank you", "thank u", "dhanyawad", "shukriya", "great help"]):
            return True, "You're very welcome! 😊 Always happy to assist you in achieving healthier soils and higher crop yields. Feel free to ask anytime. Happy farming! 🚜"

        # 5. Help / Start
        if q_clean in ["help", "start", "menu", "options"]:
            return True, (
                "Here are some helpful questions you can ask me:\n"
                "1. *'What is the fertigation schedule for Tomato during flowering?'*\n"
                "2. *'How to control thrips and upward leaf curl in Chilli?'*\n"
                "3. *'How do I prepare liquid Jeevamrutham for 1 acre?'*\n"
                "4. *'What are the symptoms and cure for Zinc deficiency in Rice?'*\n"
                "5. *'What subsidies are available for drip irrigation under PMKSY?'*"
            )

        return False, ""

    def retrieve_relevant_docs(self, query: str, context_tags: List[str] = None, top_k: int = 4) -> List[Dict[str, Any]]:
        # 1. Supabase pgvector Similarity Search (if configured)
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

        # 2. BM25 / Weighted Keyword & Context Semantic Retrieval
        stop_words = {"the", "and", "is", "for", "in", "to", "of", "a", "an", "on", "what", "how", "can", "tell", "me", "about", "give", "some", "with", "my", "our"}
        raw_words = re.findall(r"\w+", query.lower())
        query_words = [w for w in raw_words if len(w) > 2 and w not in stop_words]

        if context_tags:
            for tag in context_tags:
                if tag:
                    tag_words = [w for w in re.findall(r"\w+", tag.lower()) if len(w) > 2 and w not in stop_words]
                    query_words.extend(tag_words)

        scored_docs = []
        for doc in self.documents:
            text_lower = doc["text"].lower()
            header_lower = doc.get("header", "").lower()
            score = 0
            
            for word in query_words:
                # Higher weight if keyword matches section header
                if word in header_lower:
                    score += 8
                if word in text_lower:
                    score += min(5, text_lower.count(word))
            
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def answer_query(
        self,
        query: str,
        crop_context: str = None,
        soil_context: str = None,
        growth_stage_context: str = None,
        weather_context: str = None,
        history: List[Any] = None
    ) -> Dict[str, Any]:
        
        # 1. Handle Conversational Greetings & Small Talk naturally
        is_chat, chat_reply = self.is_conversational_query(query)
        if is_chat:
            return {
                "answer": chat_reply,
                "grounded_sources": [],
                "suggested_actions": [
                    "Ask for crop fertilizer schedules",
                    "Diagnose pest or disease symptoms",
                    "Check drip irrigation protocols"
                ]
            }

        # 2. Retrieve grounded agricultural docs
        context_tags = [crop_context, soil_context, growth_stage_context]
        relevant_docs = self.retrieve_relevant_docs(query, context_tags=context_tags, top_k=4)
        
        sources = list(set([d["source"] for d in relevant_docs])) if relevant_docs else []
        grounded_context_str = "\n\n".join([f"Source [{d['source']}]:\n{d['text']}" for d in relevant_docs]) if relevant_docs else ""

        # 3. If Gemini API Key is available, invoke LLM with strict grounding
        api_key = settings.GEMINI_API_KEY
        answer = ""
        suggested_actions = [
            "Check soil moisture before fertigation",
            "Monitor leaf undersides for sucking pests",
            "Follow stage-specific NPK dosages"
        ]

        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                system_prompt = (
                    "You are a professional, helpful, and scientifically accurate AI Agronomist & Farm Consultant.\n"
                    "Provide a well-structured, practical, and direct answer to the farmer's question. Avoid unnecessary filler or repetition.\n\n"
                    f"FARMER CONTEXT:\n"
                    f"- Crop: {crop_context or 'General'}\n"
                    f"- Soil: {soil_context or 'Standard'}\n"
                    f"- Growth Stage: {growth_stage_context or 'Active'}\n"
                    f"- Weather: {weather_context or 'Seasonal'}\n\n"
                    f"GROUNDED KNOWLEDGE BASE:\n{grounded_context_str}\n"
                )
                
                full_prompt = f"{system_prompt}\n\nFarmer's Question: {query}\n\nAgronomist Response:"
                response = model.generate_content(full_prompt)
                if response and response.text:
                    answer = response.text.strip()
            except Exception as e:
                print(f"[RAGAssistantService] Gemini API invocation notice: {e}")

        # 4. Local High-Quality Structured Synthesis if offline or API key absent
        if not answer:
            if relevant_docs:
                clean_sections = []
                for d in relevant_docs[:2]:
                    text = d["text"].strip()
                    # Clean markdown tags
                    clean_text = text.replace("# ", "").replace("## ", "**").replace("### ", "* ")
                    lines = [ln for ln in clean_text.split("\n") if ln.strip() and not ln.startswith("[")]
                    clean_sections.append("\n".join(lines[:12]))
                
                joined_body = "\n\n---\n\n".join(clean_sections)
                answer = (
                    f"### 🌾 Agronomic Guidance\n\n"
                    f"{joined_body}\n\n"
                    f"**💡 Proactive Recommendation:**\n"
                    f"- Monitor field moisture and ambient temperatures before application.\n"
                    f"- Ensure balanced nutrient splits and adhere strictly to Economic Threshold Levels (ETL) for pest treatments."
                )
            else:
                answer = (
                    f"I understand your query regarding **'{query}'**. To give you the most accurate scientific recommendation, "
                    f"could you please mention the specific crop name, soil type, or symptoms you are observing in the field? "
                    f"For example, you can ask about *'Chilli thrips control'*, *'Tomato drip fertigation'*, or *'Panchagavya preparation'*."
                )

        return {
            "answer": answer,
            "grounded_sources": sources,
            "suggested_actions": suggested_actions
        }

rag_assistant_service = RAGAssistantService()
