from __future__ import annotations # Fixes some type-hinting issues in 3.8
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from src.rag.document_processor import DocumentProcessor

class VectorStore:
    """Manage ChromaDB vector store for documents."""

    def __init__(self, collection_name: str = "property_docs"):
        self.collection_name = collection_name

        # Initialize ChromaDB (persistent)
        # Note: Ensure this path exists or is writable in your environment
        self.client = chromadb.PersistentClient(path="./chroma_db")

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Property investment documents"}
        )

    def add_documents(self, documents: List[Dict]):
        """Add documents to vector store."""
        if not documents:
            return

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for doc in documents:
            embedding = self.embedding_model.encode(doc["text"]).tolist()
            ids.append(doc["id"])
            embeddings.append(embedding)
            texts.append(doc["text"])

            processed_metadata = doc["metadata"].copy()
            if "localities" in processed_metadata and isinstance(processed_metadata["localities"], list):
                processed_metadata["localities"] = ", ".join(processed_metadata["localities"])
            metadatas.append(processed_metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print(f"Successfully added documents.")

    def search(self, query: str, n_results: int = 5,
               locality_filter: str = None, doc_type: str = None) -> List[Dict]:
        """Search for relevant documents with optional filters."""
        query_embedding = self.embedding_model.encode(query).tolist()

        chroma_where_filter = {}
        if doc_type:
            chroma_where_filter["type"] = doc_type

        # Use a larger initial fetch if we are filtering localities manually afterward
        initial_n = n_results * 5 if locality_filter else n_results
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_n,
            where=chroma_where_filter if chroma_where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        formatted_and_filtered_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            # Manual filtering for localities since they are stored as comma-separated strings
            if locality_filter:
                stored_localities = metadata.get("localities", "").lower()
                if locality_filter.lower() not in stored_localities:
                    continue

            formatted_and_filtered_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": metadata,
                "score": 1 - (results['distances'][0][i] / 2)
            })

        return formatted_and_filtered_results[:n_results]

if __name__ == "__main__":
    store = VectorStore()

    # Fixed: Define sample_docs and ensure it's passed correctly
    # sample_docs = [
    #     {
    #         "id": "test_1",
    #         "text": "Whitefield is experiencing rapid growth due to IT sector expansion.",
    #         "metadata": {
    #             "source": "test.pdf",
    #             "type": "market_news",
    #             "localities": ["Whitefield"],
    #             "chunk_index": 0
    #         }
    #     }
    # ]

    docs = DocumentProcessor().process_all_documents()
    store.add_documents(docs)

    print("\nTesting search...")
    results = store.search("property investment in Whitefield", locality_filter="Whitefield")

    for r in results:
        print(f"\nScore: {r['score']:.3f}")
        print(f"Text: {r['text'][:100]}...")