"""
analyzer.py - Refactored for Python 3.8 using Hugging Face Inference API
"""

import json
import os
import requests
import time
from typing import Dict, Any

from src.rag.retriever import DocumentRetriever
from src.models.inference import PropertyPricePredictor
from src.llm.prompts import PROMPTS

class InvestmentAnalyzer:
    """Combine ML predictions with RAG context using Hugging Face LLMs."""

    def __init__(self, hf_api_key: str = '<hugging_face_api_key>'):
        # Initialize ML and RAG components
        self.ml_predictor = PropertyPricePredictor()
        self.doc_retriever = DocumentRetriever()

        # Hugging Face Configuration
        # Defaulting to Meta-Llama-3.1-8B-Instruct (excellent for RAG/JSON)
        self.api_key = hf_api_key or os.getenv("HF_API_KEY")
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        
        if not self.api_key:
            print("Warning: HF_API_KEY not found. Using mock analysis.")

    def _query_hf_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct", # Case sensitive!
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional real estate analyst. Return only valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            # This will print the exact reason (e.g., "Model not found" or "Token invalid")
            print(f"API Error {response.status_code}: {response.text}")
            raise Exception(f"HF API Error: {response.text}")

    def analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis pipeline."""
        print(f"Analyzing property in {property_data.get('location', 'Unknown')}...")

        # Step 1: ML prediction
        ml_result = self.ml_predictor.predict(property_data)
        if not ml_result.get("success"):
            return {"error": "ML prediction failed", "details": ml_result.get("errors", [])}
        
        ml_prediction = ml_result["prediction"]

        # Step 2 & 3: RAG Retrieval
        retrieved_docs = self.doc_retriever.retrieve_for_property(property_data)
        rag_context = self.doc_retriever.format_context(retrieved_docs)

        # Step 4: Prompting
        prompt = self._prepare_prompt(property_data, ml_prediction, rag_context)

        # Step 5: LLM Synthesis
        if self.api_key:
            try:
                raw_text = self._query_hf_api(prompt)
                llm_analysis = self._parse_json_response(raw_text)
            except Exception as e:
                print(f"LLM Error: {e}")
                llm_analysis = self._get_mock_analysis(property_data, ml_prediction)
        else:
            llm_analysis = self._get_mock_analysis(property_data, ml_prediction)

        return {
            "success": True,
            "ml_prediction": ml_prediction,
            "rag_context_summary": self._summarize_rag_context(retrieved_docs),
            "investment_analysis": llm_analysis,
            "property_details": property_data
        }

    def _parse_json_response(self, text: str) -> Dict:
        """Extracts JSON from common LLM markdown wrappers."""
        try:
            # Look for JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except:
            return {"raw_output": text, "status": "parsing_failed"}

    def _prepare_prompt(self, property_data, ml_prediction, rag_context):
        # (Same logic as your original code)
        return PROMPTS["detailed"].format(
            property_details=json.dumps(property_data),
            ml_predictions=json.dumps(ml_prediction),
            rag_context=rag_context
        )

    def _get_mock_analysis(self, property_data, ml_prediction):
        # (Same logic as your original code)
        return {"recommendation": "Mock Analysis (No API Key)"}

    def _summarize_rag_context(self, retrieved_docs):
        return {k: len(v) for k, v in retrieved_docs.items()}

if __name__ == "__main__":
    # To run: python -m src.llm.analyzer
    analyzer = InvestmentAnalyzer(hf_api_key='<hugging_face_api_key>')
    test_property = {
                "location": "Indiranagar",
                "total_sqft": 1200,
                "bhk": 2,
                "bath": 2,
                "property_type": "Apartment",
                "amenities_score": 9,
            }
    print(analyzer.analyze_property(test_property))
