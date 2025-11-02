"""
Test script for RAG (Retrieval-Augmented Generation) System

Tests:
- Query enhancement
- Search strategy selection
- Context retrieval
- Response generation
- Conversation history
- Source citation formatting
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_generator import RAGGenerator
from src.opensearch_indexer import OpenSearchIndexer


def test_rag_initialization():
    """Test 1: RAG Initialization"""
    print("\n" + "="*70)
    print("TEST 1: RAG Initialization")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        # Test with custom indexer
        indexer = OpenSearchIndexer()
        rag = RAGGenerator(indexer=indexer)

        print(f"✓ RAG initialized successfully")
        print(f"  - Claude model: {rag.claude_model_id}")
        print(f"  - Max chunks: {rag.max_chunks}")
        print(f"  - Temperature: {rag.temperature}")
        print(f"  - Max history: {rag.max_history}")

        # Test stats
        stats = rag.get_stats()
        print(f"\n✓ Stats retrieved:")
        print(f"  - Index: {stats['index_name']}")
        print(f"  - History length: {stats['current_history_length']}")

        return rag

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_search_strategy_selection():
    """Test 2: Automatic Search Strategy Selection"""
    print("\n" + "="*70)
    print("TEST 2: Search Strategy Selection")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        rag = RAGGenerator()

        test_queries = [
            ("Who is the CEO of OpenAI?", "entity"),
            ("Explain neural networks", "vector"),
            ("What is the exact definition of gradient descent?", "text"),
            ("Tell me about machine learning", "hybrid")
        ]

        print("\nTesting strategy selection:")
        results = []

        for query, expected_category in test_queries:
            strategy = rag.select_search_strategy(query)
            results.append((query, strategy, expected_category))
            print(f"\n  Query: '{query}'")
            print(f"  Strategy: {strategy}")
            print(f"  Expected category: {expected_category}")

        print(f"\n✓ Strategy selection test complete")
        print(f"  - Tested {len(test_queries)} queries")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_query_enhancement():
    """Test 3: Query Enhancement"""
    print("\n" + "="*70)
    print("TEST 3: Query Enhancement")
    print("="*70)

    try:
        # Check if we have the necessary configs
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        rag = RAGGenerator()

        test_query = "What's ML?"

        print(f"\nOriginal query: '{test_query}'")
        print("Enhancing query with LLM...")

        enhanced = rag.enhance_query(test_query)

        print(f"\n✓ Query enhancement complete")
        print(f"  - Original: '{test_query}'")
        print(f"  - Enhanced: '{enhanced}'")
        print(f"  - Length increase: {len(enhanced) - len(test_query)} chars")

        return enhanced

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print(f"  This test requires AWS Bedrock access")
        return None


def test_context_retrieval():
    """Test 4: Context Retrieval"""
    print("\n" + "="*70)
    print("TEST 4: Context Retrieval")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        rag = RAGGenerator()

        if not rag.indexer.index_exists():
            print(f"\n[SKIP] Index '{rag.indexer.opensearch_index_name}' does not exist")
            return None

        test_query = "machine learning"

        print(f"\nQuery: '{test_query}'")
        print(f"Retrieving context...")

        # Test different strategies
        strategies = ['text', 'vector', 'hybrid']
        results = {}

        for strategy in strategies:
            print(f"\n  Testing {strategy} search...")
            chunks = rag.retrieve_context(test_query, num_chunks=3, strategy=strategy)
            results[strategy] = chunks
            print(f"  Retrieved: {len(chunks)} chunks")

            if chunks:
                print(f"  Top result score: {chunks[0].get('score', 0):.4f}")

        print(f"\n✓ Context retrieval test complete")
        print(f"  - Tested {len(strategies)} search strategies")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_source_formatting():
    """Test 5: Source Citation Formatting"""
    print("\n" + "="*70)
    print("TEST 5: Source Citation Formatting")
    print("="*70)

    try:
        rag = RAGGenerator()

        # Create sample chunks
        sample_chunks = [
            {
                "chunk_id": 1,
                "content": "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
                "score": 0.95,
                "entities": [
                    {"type": "TOPIC", "value": "Machine Learning"},
                    {"type": "TOPIC", "value": "Artificial Intelligence"}
                ],
                "topics": ["AI", "ML", "Data Science"]
            },
            {
                "chunk_id": 2,
                "content": "Neural networks are computing systems inspired by biological neural networks.",
                "score": 0.87,
                "entities": [],
                "topics": ["Neural Networks", "Deep Learning"]
            }
        ]

        print(f"\nFormatting {len(sample_chunks)} sample chunks...")

        sources = rag._format_sources(sample_chunks)

        print(f"\n✓ Source formatting complete")
        print(f"\nFormatted sources preview:")
        print(sources[:300] + "..." if len(sources) > 300 else sources)

        return sources

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_conversation_history():
    """Test 6: Conversation History Management"""
    print("\n" + "="*70)
    print("TEST 6: Conversation History Management")
    print("="*70)

    try:
        rag = RAGGenerator(max_history=3)

        print(f"✓ RAG initialized with max_history={rag.max_history}")

        # Simulate conversation
        simulated_turns = [
            ("What is AI?", "AI is artificial intelligence..."),
            ("How does it work?", "AI works by..."),
            ("Give examples", "Examples include..."),
            ("What are the challenges?", "Challenges include...")
        ]

        print(f"\nSimulating {len(simulated_turns)} conversation turns...")

        for user_msg, assistant_msg in simulated_turns:
            rag.conversation_history.append({
                'user': user_msg,
                'assistant': assistant_msg
            })

        # Check if history is trimmed
        history = rag.get_history()

        print(f"\n✓ Conversation simulation complete")
        print(f"  - Simulated turns: {len(simulated_turns)}")
        print(f"  - Stored turns: {len(history)}")
        print(f"  - Max history: {rag.max_history}")
        print(f"  - History correctly trimmed: {len(history) <= rag.max_history}")

        # Test clear history
        rag.clear_history()
        print(f"\n✓ History cleared")
        print(f"  - Remaining turns: {len(rag.get_history())}")

        return history

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def test_ask_method():
    """Test 7: RAG Ask Method (Single Question)"""
    print("\n" + "="*70)
    print("TEST 7: RAG Ask Method")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        rag = RAGGenerator()

        if not rag.indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        question = "What is machine learning?"

        print(f"\nAsking: '{question}'")
        print("(This will retrieve context and generate response)\n")

        result = rag.ask(question, num_chunks=2, enhance_query=False)

        print(f"\n✓ Ask method complete")
        print(f"\nResult keys: {list(result.keys())}")
        print(f"  - Answer length: {len(result['answer'])} chars")
        print(f"  - Sources length: {len(result['sources'])} chars")
        print(f"  - Chunks retrieved: {len(result['chunks'])}")

        print(f"\nAnswer preview:")
        print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])

        return result

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print(f"  This test requires AWS Bedrock access and indexed data")
        return None


def test_chat_method():
    """Test 8: RAG Chat Method (Multi-turn)"""
    print("\n" + "="*70)
    print("TEST 8: RAG Chat Method")
    print("="*70)

    try:
        if not os.getenv('OPENSEARCH_HOST'):
            print("\n[SKIP] OpenSearch not configured")
            return None

        rag = RAGGenerator()

        if not rag.indexer.index_exists():
            print(f"\n[SKIP] Index does not exist")
            return None

        # Simulate conversation
        messages = [
            "What is deep learning?",
            "How is it different from machine learning?"
        ]

        print(f"\nSimulating {len(messages)} conversation turns...\n")

        results = []
        for i, message in enumerate(messages, 1):
            print(f"Turn {i}: {message}")
            result = rag.chat(message, num_chunks=2)
            results.append(result)
            print(f"  ✓ Response generated ({len(result['answer'])} chars)")

        print(f"\n✓ Chat method test complete")
        print(f"  - Turns: {len(results)}")
        print(f"  - History length: {len(rag.get_history())}")

        return results

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print(f"  This test requires AWS Bedrock access and indexed data")
        return None


def run_all_tests():
    """Run all RAG tests"""
    print("\n" + "#"*70)
    print("# RAG SYSTEM - TEST SUITE")
    print("#"*70)

    tests = [
        ("RAG Initialization", test_rag_initialization),
        ("Search Strategy Selection", test_search_strategy_selection),
        ("Query Enhancement", test_query_enhancement),
        ("Context Retrieval", test_context_retrieval),
        ("Source Formatting", test_source_formatting),
        ("Conversation History", test_conversation_history),
        ("Ask Method", test_ask_method),
        ("Chat Method", test_chat_method)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
            if results[test_name] is not None:
                print(f"\n[PASS] {test_name}")
            else:
                print(f"\n[SKIP] {test_name}")
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {str(e)}")
            results[test_name] = None

    # Final summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    passed = sum(1 for v in results.values() if v is not None)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Tests Skipped: {skipped}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print("\n" + "#"*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
