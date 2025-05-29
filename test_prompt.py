from src.validation.schema import MathQuery
from src.knowledge_base.vectorstore import MathKnowledgeBase
from src.mcp.solution_verifier import MCPSolutionVerifier
from src.feedback.collector import FeedbackCollector

def process_math_query(query: str, force_mcp: bool = False):
    """Process a math query through the entire pipeline."""
    print(f"\n🔍 Processing query: {query}\n")
    
    # 1. Validate the query
    try:
        validated_query = MathQuery(query=query)
        print("✅ Query validation passed\n")
    except ValueError as e:
        print(f"❌ Query validation failed: {e}\n")
        return

    # 2. Check knowledge base first (using MATH dataset)
    kb = MathKnowledgeBase()
    kb.load_data()  # This will load the MATH dataset from Hugging Face
    kb_results = kb.search(query)
    
    # First try knowledge base if not forcing MCP
    if not force_mcp and kb_results and kb_results[0]['similarity_score'] > 0.8:
        print(f"📚 Found similar problem in knowledge base (similarity: {kb_results[0]['similarity_score']:.2f})")
        print("\nQuestion:", kb_results[0]['question'])
        print("\nSolution:", kb_results[0]['solution'])
        print("\nCategory:", kb_results[0]['category'])
        
        # Collect feedback
        feedback = FeedbackCollector()
        feedback.collect(query, kb_results[0]['solution'])
        return

    # 3. No good match in knowledge base or forcing MCP, use MCP
    print("\n🤖 Using MCP to solve...")
    mcp = MCPSolutionVerifier()
    solution = mcp.generate_solution_with_verification(query)
    
    if solution.startswith("Could not"):
        print(f"\n❌ {solution}")
    else:
        print("\n✅ Solution generated:")
        print(solution)
        
        # Collect feedback
        feedback = FeedbackCollector()
        feedback.collect(query, solution)

if __name__ == "__main__":
    # Test query
    query = "Prove that the square root of 2 is irrational"
    
    print("First, let's try using the knowledge base:")
    process_math_query(query)
    
    print("\n" + "="*80 + "\n")
    
    print("Now, let's force it to use MCP:")
    process_math_query(query, force_mcp=True) 