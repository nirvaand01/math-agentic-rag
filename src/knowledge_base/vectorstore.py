from typing import List, Dict, Any
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

class MathKnowledgeBase:
    """Knowledge base for math problems using Qdrant vector store."""
    
    def __init__(self):
        """Initialize the knowledge base."""
        # Initialize the embedding model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Qdrant client (in-memory)
        self.client = QdrantClient(":memory:")
        self.collection_name = "math_problems"
        
        # Initialize the vector store
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize the vector store with math problems."""
        print("Loading Berkeley MATH dataset from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("EleutherAI/hendrycks_math", "algebra")
        problems = dataset["train"][:1000]  # Load first 1000 problems for faster startup
        
        # Create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        
        # Prepare problems for vectorization
        texts = []
        metadata = []
        
        for problem in problems:
            try:
                texts.append(f"{problem['problem']}\n{problem['solution']}")
                metadata.append({
                    "question": problem["problem"],
                    "solution": problem["solution"],
                    "answer": problem.get("answer", ""),
                    "category": "algebra"
                })
            except Exception as e:
                print(f"Skipping problem: {e}")
                continue
        
        print(f"Loading {len(texts)} problems into vector store...")
        
        # Vectorize in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            # Get embeddings for batch
            embeddings = self.model.encode(batch_texts)
            
            # Create points
            points = [
                models.PointStruct(
                    id=idx + i,
                    vector=embedding.tolist(),
                    payload=meta
                )
                for idx, (embedding, meta) in enumerate(zip(embeddings, batch_metadata))
            ]
            
            # Upload batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Uploaded {i + len(batch_texts)}/{len(texts)} problems")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar math problems.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of similar problems with their solutions
        """
        # Get query embedding
        query_vector = self.model.encode(query).tolist()
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for hit in results:
            result = hit.payload.copy()  # Copy to avoid modifying the original
            result["similarity_score"] = float(hit.score) if hasattr(hit, "score") else 0.0
            formatted_results.append(result)
            
        return formatted_results 