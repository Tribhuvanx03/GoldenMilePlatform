"""
retriever.py - Retrieve relevant documents for property analysis
Simple retriever with locality filtering.
"""

from src.rag.vector_store import VectorStore
from typing import List, Dict


class DocumentRetriever:
    """Retrieve relevant documents for property investment analysis."""

    def __init__(self):
        self.vector_store = VectorStore()

    def retrieve_for_property(self, property_data: Dict, n_results: int = 5) -> Dict:
        """
        Retrieve relevant documents for a specific property.

        Args:
            property_data: Dict with at least 'location' key
            n_results: Number of documents to retrieve per category

        Returns:
            Dict with categorized documents
        """
        location = property_data.get("location", "").lower()

        # Retrieve different types of documents
        regulatory_docs = self.vector_store.search(
            query = f"RERA stamp duty registration legal compliance Bangalore {location}",
            n_results=n_results,
            locality_filter=location,
            doc_type="regulatory"
        )

        market_docs = self.vector_store.search(
            query = f"property prices growth infrastructure development {location} Bangalore real estate market",
            n_results=n_results,
            locality_filter=location,
            doc_type="market_news"
        )

        locality_docs = self.vector_store.search(
            query = f"{location} area locality amenities schools hospitals transport connectivity",
            n_results=n_results,
            locality_filter=location,
            doc_type="locality_profile"
        )

        # If no locality-specific docs found, get general docs
        if not any([regulatory_docs, market_docs, locality_docs]):
            general_docs = self.vector_store.search(
                query="Bangalore real estate property investment",
                n_results=n_results
            )
        else:
            general_docs = []

        return {
            "regulatory": regulatory_docs,
            "market_news": market_docs,
            "locality_profile": locality_docs,
            "general": general_docs,
            "query_location": location
        }

    def format_context(self, retrieved_docs: Dict) -> str:
        """Format retrieved documents into context string for LLM."""
        context_parts = []

        for category, docs in retrieved_docs.items():
            if category == "query_location": # Skip the query_location entry
                continue
            if docs:
                context_parts.append(f"\n=== {category.upper().replace('_', ' ')} ===\n")
                for i, doc in enumerate(docs, 1):
                    context_parts.append(f"Document {i} (Score: {doc['score']:.2f}):")
                    context_parts.append(f"Source: {doc['metadata']['source']}")
                    context_parts.append(f"Text: {doc['text']}\n")

        return "\n".join(context_parts)


# Simple usage
if __name__ == "__main__":
    retriever = DocumentRetriever()

    # Test with sample property
    property_info = {
        "location": "Yelahanka",
        "total_sqft": 1500,
        "bhk": 3
    }

    docs = retriever.retrieve_for_property(property_info)
    context = retriever.format_context(docs)

    print(f"Retrieved {sum(len(d) for k,d in docs.items() if k != 'query_location')} documents")
    print("\nContext preview:")
    print(context[:500] + "...")
