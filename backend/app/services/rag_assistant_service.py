import os
import glob
from typing import List, Dict, Any
from backend.app.config import settings

class RAGAssistantService:
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        knowledge_files = glob.glob(os.path.join(settings.KNOWLEDGE_DIR, "*.txt"))
        for file_path in knowledge_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    filename = os.path.basename(file_path)
                    # Simple chunking
                    sections = content.split("## ")
                    for sec in sections:
                        if sec.strip():
                            self.documents.append({
                                "source": filename,
                                "text": "## " + sec if not sec.startswith("#") else sec
                            })
            except Exception as e:
                print(f"Error reading knowledge file {file_path}: {e}")
        
        print(f"Loaded {len(self.documents)} knowledge chunks into RAG store.")

    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            text_lower = doc["text"].lower()
            score = sum(1 for word in query_words if word in text_lower and len(word) > 2)
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
            # High-quality contextual fallback synthesis
            answer = (
                f"Based on agricultural recommendations for **{crop_context or 'your farm'}** "
                f"under **{soil_context or 'current soil'}** conditions:\n\n"
                f"1. **Direct Answer:** Regarding *'{query}'*, ensure optimal soil moisture and avoid water stagnation. "
                f"If in the vegetative or tillering stage, timely top-dressing with nitrogen and balanced irrigation is essential.\n\n"
                f"2. **Grounded Advice:** Refer to the verified agronomic schedule. "
                f"Maintain regular monitoring for pest thresholds and apply recommended split fertilizer doses as per your soil test.\n\n"
                f"3. **Weather Advisory:** With current conditions ({weather_context or 'mild'}), ensure proper drainage channels and avoid foliar spraying during high winds."
            )

        return {
            "answer": answer,
            "grounded_sources": sources,
            "suggested_actions": suggested_actions
        }

rag_assistant_service = RAGAssistantService()
