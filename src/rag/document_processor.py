"""
document_processor.py - Process PDF documents for RAG
Simple PDF text extraction with metadata tagging.
"""

import PyPDF2
import os
from typing import List, Dict, Tuple
import re
import pandas as pd


class DocumentProcessor:
    """Process PDF documents into text chunks with metadata."""
    
    def __init__(self, knowledge_base_path: str = "/project/workspace/GoldenMile/data/knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text
    
    def detect_locality(self, text: str) -> List[str]:
        """Extract Bangalore localities mentioned in text."""

        bangalore_localities = (list(pd.read_csv("/project/workspace/GoldenMile/data/processed/BangaloreDataMod.csv")['location']))
        bangaore_local = set(bangalore_localities)
        # #print(len(bangaore_local))
        bangalore_localities = list(bangaore_local)
        bangalore_localities.remove('other')
        bangalore_localities = [item.lower() for item in bangalore_localities]
        

        # Common Bangalore localities
        # bangalore_localities = [
        #      "whitefield", "indiranagar", "koramangala", "jayanagar", "hsr layout",
        #      "marathahalli", "bellandur", "electronic city", "hebbal", "yelahanka",
        #      "rajaji nagar", "malleshwaram", "basavanagudi", "btm layout", 
        #      "sanjay nagar", "frazer town", "richmond town", "c v raman nagar",
        #      "k r puram", "sarjapur", "banashankari", "jp nagar", "bannerghatta",
        #      "vijayanagar", "magadi road", "mysore road", "old airport road"
        # ]
        
        found_localities = []
        text_lower = text.lower()
        
        for locality in bangalore_localities:
            if locality in text_lower:
                found_localities.append(locality.title())
        
        return found_localities
    
    def detect_document_type(self, filename: str, text: str) -> str:
        """Determine document type based on filename and content."""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if "rera" in filename_lower or "rera" in text_lower:
            return "regulatory"
        elif "registration" in filename_lower or "stamp" in filename_lower:
            return "regulatory"
        elif "legal" in filename_lower or "compliance" in filename_lower:
            return "regulatory"
        elif "report" in filename_lower or "market" in filename_lower:
            return "market_news"
        elif "infrastructure" in filename_lower:
            return "market_news"
        elif "locality" in filename_lower:
            return "locality_profile"
        else:
            # Guess from content
            if "rbi" in text_lower or "government" in text_lower:
                return "regulatory"
            elif "price" in text_lower or "growth" in text_lower:
                return "market_news"
            else:
                return "general"
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def process_all_documents(self) -> List[Dict]:
        """Process all PDFs in knowledge base."""
        documents = []
        
        # Walk through knowledge base directory
        for root, dirs, files in os.walk(self.knowledge_base_path):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    
                    # Extract text
                    text = self.extract_text_from_pdf(pdf_path)
                    if not text:
                        continue
                    
                    # Detect metadata
                    localities = self.detect_locality(text)
                    doc_type = self.detect_document_type(file, text)
                    
                    # Create chunks
                    chunks = self.chunk_text(text)
                    
                    # Create document entries
                    for i, chunk in enumerate(chunks):
                        doc_entry = {
                            "id": f"{file}_{i}",
                            "text": chunk,
                            "metadata": {
                                "source": file,
                                "type": doc_type,
                                "localities": localities,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        }
                        documents.append(doc_entry)
                    
                    print(f"  → {len(chunks)} chunks, localities: {localities}")
        
        print(f"\nTotal documents processed: {len(documents)}")
        return documents


# Simple usage example
if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.process_all_documents()
    
    # Show sample
    if docs:
        print("\nSample document:")
        print(f"ID: {docs[101]['id']}")
        print(f"Type: {docs[101]['metadata']['type']}")
        print(f"Localities: {docs[101]['metadata']['localities']}")
        print(f"Text preview: {docs[101]['text'][:100]}...")