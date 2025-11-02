"""
RAG (Retrieval-Augmented Generation) Example

Demonstrates the RAG system with:
- Single question answering
- Multi-turn conversations
- Source citations
- Query enhancement
"""

from rag_generator import RAGGenerator
from opensearch_indexer import OpenSearchIndexer


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"{'='*80}\n")


def print_answer(result: dict):
    """
    Print RAG response in formatted way.

    Args:
        result: Result dictionary from RAG
    """
    print("\n" + "-"*80)
    print("ANSWER:")
    print("-"*80)
    print(result['answer'])

    if result.get('sources'):
        print("\n" + "-"*80)
        print("SOURCES:")
        print("-"*80)
        print(result['sources'])

    print("\n" + "-"*80)


def demo_single_question():
    """Demonstrate single question-answering."""
    print_separator("Demo 1: Single Question Answering")

    # Initialize RAG
    print("Initializing RAG system...")
    rag = RAGGenerator()

    # Check if index exists
    if not rag.indexer.index_exists():
        print(f"\n⚠️  Error: Index '{rag.indexer.opensearch_index_name}' does not exist!")
        print("Please ensure:")
        print("  1. OpenSearch is configured")
        print("  2. Index is created (run scripts/create_opensearch_index.py)")
        print("  3. Documents are indexed (run src/example_opensearch.py)")
        return

    stats = rag.get_stats()
    print(f"✓ RAG initialized")
    print(f"  - Model: {stats['claude_model']}")
    print(f"  - Max chunks: {stats['max_chunks']}")
    print(f"  - Temperature: {stats['temperature']}")
    print(f"  - Index: {stats['index_name']}")

    # Ask a question
    question = "What is machine learning?"

    result = rag.ask(question, num_chunks=3, enhance_query=True, include_sources=True)
    print_answer(result)


def demo_multi_turn_conversation():
    """Demonstrate multi-turn conversation with history."""
    print_separator("Demo 2: Multi-Turn Conversation")

    # Initialize RAG
    print("Initializing RAG system...")
    rag = RAGGenerator()

    if not rag.indexer.index_exists():
        print(f"\n⚠️  Index not found. Skipping demo.")
        return

    print(f"✓ RAG initialized for conversation mode\n")

    # Conversation flow
    conversation = [
        "What is deep learning?",
        "How does it differ from traditional machine learning?",
        "Can you give me examples of applications?"
    ]

    print("Starting conversation with 3 turns...\n")

    for i, question in enumerate(conversation, 1):
        print(f"\n{'#'*80}")
        print(f"TURN {i}")
        print(f"{'#'*80}")
        print(f"\nUser: {question}")

        result = rag.chat(question, num_chunks=3)

        print(f"\nAssistant: {result['answer']}")

        # Show sources for first turn only to keep output manageable
        if i == 1:
            print(f"\n[Sources for Turn 1]")
            print(result['sources'][:300] + "..." if len(result['sources']) > 300 else result['sources'])

    # Show conversation history
    print(f"\n{'#'*80}")
    print("CONVERSATION HISTORY")
    print(f"{'#'*80}\n")

    history = rag.get_history()
    print(f"Total turns: {len(history)}\n")

    for i, turn in enumerate(history, 1):
        print(f"Turn {i}:")
        print(f"  User: {turn['user']}")
        print(f"  Assistant: {turn['assistant'][:100]}...")
        print()


def demo_query_enhancement():
    """Demonstrate query enhancement."""
    print_separator("Demo 3: Query Enhancement")

    print("Initializing RAG system...")
    rag = RAGGenerator()

    if not rag.indexer.index_exists():
        print(f"\n⚠️  Index not found. Skipping demo.")
        return

    print(f"✓ RAG initialized\n")

    # Test query enhancement
    test_queries = [
        "What's ML?",
        "Tell me about neural nets",
        "Who invented AI?"
    ]

    print("Testing query enhancement on various queries:\n")

    for query in test_queries:
        print(f"Original: '{query}'")
        enhanced = rag.enhance_query(query)
        print(f"Enhanced: '{enhanced}'")
        print()


def demo_search_strategies():
    """Demonstrate automatic search strategy selection."""
    print_separator("Demo 4: Automatic Search Strategy Selection")

    print("Initializing RAG system...")
    rag = RAGGenerator()

    if not rag.indexer.index_exists():
        print(f"\n⚠️  Index not found. Skipping demo.")
        return

    print(f"✓ RAG initialized\n")

    # Test different query types
    test_queries = [
        ("Who created Python?", "entity-focused"),
        ("Explain the concept of overfitting", "semantic/conceptual"),
        ("What is the exact term for gradient descent?", "keyword-focused"),
        ("What are neural networks?", "general")
    ]

    print("Testing automatic strategy selection:\n")

    for query, query_type in test_queries:
        print(f"Query: '{query}'")
        print(f"Type: {query_type}")
        strategy = rag.select_search_strategy(query)
        print(f"Selected Strategy: {strategy}")
        print()


def interactive_mode():
    """Interactive RAG session."""
    print_separator("Interactive RAG Mode")

    print("Initializing RAG system...")
    rag = RAGGenerator()

    if not rag.indexer.index_exists():
        print(f"\n⚠️  Index not found. Cannot start interactive mode.")
        return

    print(f"✓ RAG initialized")
    print(f"\nCommands:")
    print(f"  - Type your question to ask")
    print(f"  - '/clear' to clear history")
    print(f"  - '/history' to show conversation history")
    print(f"  - '/quit' to exit")
    print(f"\n{'='*80}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input == '/quit':
                print("\nGoodbye!")
                break

            if user_input == '/clear':
                rag.clear_history()
                continue

            if user_input == '/history':
                history = rag.get_history()
                if not history:
                    print("No conversation history yet.")
                else:
                    print(f"\nConversation History ({len(history)} turns):")
                    for i, turn in enumerate(history, 1):
                        print(f"\n{i}. User: {turn['user']}")
                        print(f"   Assistant: {turn['assistant'][:100]}...")
                continue

            # Ask question
            result = rag.chat(user_input, num_chunks=3)
            print(f"\nAssistant: {result['answer']}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main function to run RAG demos."""
    print_separator("RAG (Retrieval-Augmented Generation) Demo")

    print("This demo showcases the RAG system capabilities:")
    print("  1. Single question answering with source citations")
    print("  2. Multi-turn conversations with context")
    print("  3. Query enhancement using LLM")
    print("  4. Automatic search strategy selection")
    print("  5. Interactive mode")

    print("\nSelect demo to run:")
    print("  1 - Single Question Answering")
    print("  2 - Multi-Turn Conversation")
    print("  3 - Query Enhancement")
    print("  4 - Search Strategy Selection")
    print("  5 - Interactive Mode")
    print("  6 - Run All Demos (1-4)")

    try:
        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            demo_single_question()
        elif choice == '2':
            demo_multi_turn_conversation()
        elif choice == '3':
            demo_query_enhancement()
        elif choice == '4':
            demo_search_strategies()
        elif choice == '5':
            interactive_mode()
        elif choice == '6':
            demo_single_question()
            demo_multi_turn_conversation()
            demo_query_enhancement()
            demo_search_strategies()
        else:
            print("Invalid choice. Running demo 1 by default.")
            demo_single_question()

        print_separator("Demo Complete!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("  1. .env file is properly configured")
        print("  2. AWS credentials are set up")
        print("  3. OpenSearch index exists and has data")
        print("  4. Bedrock Claude access is enabled in your AWS account")


if __name__ == "__main__":
    main()
