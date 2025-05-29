from typing import List, Dict, Any
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

class KnowledgeBase:
    """Class to manage the knowledge base for math problems and solutions."""
    
    def __init__(self, collection_name: str = "math_qa"):
        """Initialize the knowledge base."""
        # Initialize sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize Qdrant client
        self.client = QdrantClient(":memory:")  # Use in-memory storage for testing
        
        # Create collection if it doesn't exist
        self.collection_name = collection_name
        self.create_collection()
        
        # Load initial Q&A pairs if available
        self.load_initial_data()
    
    def create_collection(self):
        """Create vector collection if it doesn't exist."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Dimension of sentence-transformer embeddings
                    distance=models.Distance.COSINE
                )
            )
        except Exception as e:
            print(f"Collection might already exist: {str(e)}")
    
    def load_initial_data(self):
        """Load initial Q&A pairs from JSON file if available."""
        data_file = "data/math_qa_pairs.json"
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                qa_pairs = json.load(f)
            
            # Add each Q&A pair to the vector store
            for i, qa in enumerate(qa_pairs):
                self.add_qa_pair(qa["question"], qa["answer"], qa.get("metadata", {}), i)
    
    def add_qa_pair(self, question: str, answer: str, metadata: Dict[str, Any], id: int):
        """Add a Q&A pair to the knowledge base."""
        # Generate embedding for the question
        embedding = self.model.encode(question)
        
        # Add point to the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    vector=embedding.tolist(),
                    payload={
                        "question": question,
                        "answer": answer,
                        **metadata
                    }
                )
            ]
        )
    
    def search_similar_questions(self, question: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar questions in the knowledge base."""
        # Generate embedding for the query
        query_vector = self.model.encode(question)
        
        # Search for similar questions
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        # Format results
        similar_qa = []
        for result in results:
            similar_qa.append({
                "question": result.payload["question"],
                "answer": result.payload["answer"],
                "score": result.score,
                **{k: v for k, v in result.payload.items() if k not in ["question", "answer"]}
            })
        
        return similar_qa 