 e 2x + 3 = 7",
        "Calculate the integral of sin(x)"
    ]
    
    for query in valid_queries:
        assert MathQuery(query=query)
    
    # Invalid queries
    invalid_queries = [
        "What is the weather today?",
        "Tell me a joke",
        "ab"  # too short
    ]
    
    for query in invalid_queries:
        with pytest.raises(ValueError):
            MathQuery(query=query)

def test_knowledge_base():
    """Test knowledge base retrieval."""
    kb = MathKnowledgeBase()
    kb.load_data("data/math_qa.json")
    
    # Test search
    results = kb.search("Find the derivative of x^2")
    assert len(results) > 0
    assert "derivative" in results[0]["question"].lower()
    assert isinstance(results[0]["similarity_score"], float)

def test_web_search():
    """Test web search functionality."""
    searcher = MathWebSearcher()
    results = searcher.search("calculus integration by parts formula")
    assert len(results) > 0
    assert isinstance(results[0]["title"], str)
    assert isinstance(results[0]["content"], str)

def test_solution_generation():
    """Test solution generation."""
    generator = MathSolutionGenerator()
    solution = generator.generate(
        "Find the derivative of x^2",
        context="The derivative of x^n is nx^(n-1)"
    )
    assert solution
    assert "2x" in solution.lower()

def test_feedback_collection():
    """Test feedback collection."""
    collector = FeedbackCollector()
    feedback = collector.add_feedback(
        question="What is 2+2?",
        solution="4",
        rating="positive",
        feedback_text="Clear and correct"
    )
    assert feedback["rating"] == "positive"
    assert feedback["feedback_text"] == "Clear and correct"

if __name__ == "__main__":
    pytest.main([__file__]) 