"""
Interactive RAG Command-Line Interface

Provides a user-friendly CLI for RAG-based question answering with:
- Conversation mode with history
- Special commands for control
- Colored output for better readability
- Source citations
- Query statistics
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_generator import RAGGenerator


def print_header():
    """Print application header."""
    print("\n" + "="*80)
    print("  RAG INTERACTIVE MODE - Retrieval-Augmented Generation")
    print("="*80)


def print_help():
    """Print help message with available commands."""
    print("\nAvailable Commands:")
    print("  /help       - Show this help message")
    print("  /clear      - Clear conversation history")
    print("  /history    - Show conversation history")
    print("  /stats      - Show RAG system statistics")
    print("  /chunks N   - Set number of chunks to retrieve (default: 5)")
    print("  /temp N     - Set temperature (0.0-1.0, default: 0.7)")
    print("  /quit       - Exit the application")
    print("\nOr simply type your question to ask the RAG system.")


def print_stats(rag: RAGGenerator):
    """Print RAG system statistics."""
    stats = rag.get_stats()
    print("\n" + "-"*80)
    print("RAG SYSTEM STATISTICS")
    print("-"*80)
    print(f"Model:            {stats['claude_model']}")
    print(f"Max Chunks:       {stats['max_chunks']}")
    print(f"Temperature:      {stats['temperature']}")
    print(f"Max History:      {stats['max_history']}")
    print(f"Current History:  {stats['current_history_length']} turns")
    print(f"OpenSearch Host:  {stats['opensearch_host']}")
    print(f"Index Name:       {stats['index_name']}")
    print("-"*80)


def print_history(rag: RAGGenerator):
    """Print conversation history."""
    history = rag.get_history()

    if not history:
        print("\nNo conversation history yet.")
        return

    print("\n" + "-"*80)
    print(f"CONVERSATION HISTORY ({len(history)} turns)")
    print("-"*80)

    for i, turn in enumerate(history, 1):
        print(f"\n[Turn {i}]")
        print(f"You: {turn['user']}")
        print(f"Assistant: {turn['assistant'][:150]}...")
        if len(turn['assistant']) > 150:
            print(f"           ({len(turn['assistant']) - 150} more characters)")

    print("-"*80)


def print_answer(result: dict, show_sources: bool = True):
    """
    Print answer with formatting.

    Args:
        result: Result dictionary from RAG
        show_sources: Whether to display sources
    """
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['answer'])

    if show_sources and result.get('sources'):
        print("\n" + "-"*80)
        print("SOURCES:")
        print("-"*80)
        # Truncate very long sources
        sources = result['sources']
        if len(sources) > 1000:
            print(sources[:1000])
            print(f"\n... ({len(sources) - 1000} more characters)")
        else:
            print(sources)

    print("="*80 + "\n")


def handle_command(command: str, rag: RAGGenerator) -> bool:
    """
    Handle special commands.

    Args:
        command: Command string
        rag: RAG instance

    Returns:
        True if should continue, False if should quit
    """
    parts = command.split()
    cmd = parts[0].lower()

    if cmd == '/quit' or cmd == '/exit':
        return False

    elif cmd == '/help':
        print_help()

    elif cmd == '/clear':
        rag.clear_history()
        print("\n✓ Conversation history cleared.")

    elif cmd == '/history':
        print_history(rag)

    elif cmd == '/stats':
        print_stats(rag)

    elif cmd == '/chunks':
        if len(parts) < 2:
            print(f"\nCurrent max chunks: {rag.max_chunks}")
            print("Usage: /chunks <number>")
        else:
            try:
                num = int(parts[1])
                if 1 <= num <= 20:
                    rag.max_chunks = num
                    print(f"\n✓ Max chunks set to {num}")
                else:
                    print("\n✗ Please specify a number between 1 and 20")
            except ValueError:
                print("\n✗ Invalid number")

    elif cmd == '/temp' or cmd == '/temperature':
        if len(parts) < 2:
            print(f"\nCurrent temperature: {rag.temperature}")
            print("Usage: /temp <0.0-1.0>")
        else:
            try:
                temp = float(parts[1])
                if 0.0 <= temp <= 1.0:
                    rag.temperature = temp
                    print(f"\n✓ Temperature set to {temp}")
                else:
                    print("\n✗ Temperature must be between 0.0 and 1.0")
            except ValueError:
                print("\n✗ Invalid number")

    else:
        print(f"\n✗ Unknown command: {cmd}")
        print("Type /help for available commands")

    return True


def main():
    """Main interactive loop."""
    print_header()

    try:
        # Initialize RAG
        print("\nInitializing RAG system...")
        rag = RAGGenerator()

        # Check if index exists
        if not rag.indexer.index_exists():
            print(f"\n✗ Error: Index '{rag.indexer.opensearch_index_name}' does not exist!")
            print("\nPlease ensure:")
            print("  1. OpenSearch is configured in .env")
            print("  2. Index is created (run: python scripts/create_opensearch_index.py)")
            print("  3. Documents are indexed (run: python src/example_opensearch.py)")
            return

        print("✓ RAG system initialized successfully!")

        # Show initial stats
        print_stats(rag)
        print_help()

        print("\n" + "="*80)
        print("Ready! Type your question or use /help for commands.")
        print("="*80 + "\n")

        # Main interaction loop
        while True:
            try:
                user_input = input("You: ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if not handle_command(user_input, rag):
                        print("\nThank you for using the RAG system. Goodbye!")
                        break
                    continue

                # Ask RAG question
                print("\n⏳ Processing...")

                result = rag.chat(user_input, num_chunks=rag.max_chunks)

                print_answer(result, show_sources=True)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                confirm = input("Do you want to quit? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("\nGoodbye!")
                    break
                else:
                    print("\nContinuing...\n")

            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Please try again or type /quit to exit.\n")

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        print("\nPlease ensure:")
        print("  1. .env file is properly configured")
        print("  2. AWS credentials are set up")
        print("  3. OpenSearch is accessible")
        print("  4. Bedrock Claude access is enabled")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
