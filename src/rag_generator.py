"""
RAG (Retrieval-Augmented Generation) System

Combines semantic search with LLM generation to answer questions based on indexed documents.

Features:
- Automatic search strategy selection
- Query enhancement using LLM
- Conversation history management
- Source citations
- AWS Bedrock Claude integration
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from opensearch_indexer import OpenSearchIndexer
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False

# Load environment variables
load_dotenv()


class RAGGenerator:
    """
    Retrieval-Augmented Generation system using OpenSearch and AWS Bedrock Claude.

    Provides intelligent question-answering with source citations,
    conversation history, and automatic search strategy selection.
    """

    def __init__(
        self,
        indexer: Optional[OpenSearchIndexer] = None,
        claude_model_id: Optional[str] = None,
        max_chunks: Optional[int] = None,
        temperature: Optional[float] = None,
        max_history: Optional[int] = None,
        bedrock_region: Optional[str] = None
    ):
        """
        Initialize RAG Generator.

        Args:
            indexer: OpenSearchIndexer instance (will create new if None)
            claude_model_id: Bedrock Claude model ID
            max_chunks: Maximum chunks to retrieve for context
            temperature: LLM temperature (0.0-1.0)
            max_history: Maximum conversation turns to keep
            bedrock_region: AWS region for Bedrock
        """
        # Check dependencies
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required. Install with: pip install boto3")

        if not INDEXER_AVAILABLE:
            raise ImportError(
                "OpenSearchIndexer is required. Ensure opensearch_indexer.py is available."
            )

        # Initialize indexer
        if indexer is None:
            self.indexer = OpenSearchIndexer()
        else:
            self.indexer = indexer

        # Configuration
        self.claude_model_id = claude_model_id or os.getenv(
            'BEDROCK_CLAUDE_MODEL_ID',
            'anthropic.claude-3-sonnet-20240229-v1:0'
        )
        self.max_chunks = max_chunks or int(os.getenv('RAG_MAX_CHUNKS', '5'))
        self.temperature = temperature or float(os.getenv('RAG_TEMPERATURE', '0.7'))
        self.max_history = max_history or int(os.getenv('RAG_MAX_HISTORY', '10'))
        self.bedrock_region = bedrock_region or os.getenv('BEDROCK_REGION', 'us-east-1')

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Bedrock client cache
        self._bedrock_runtime = None

    def _get_bedrock_runtime_client(self):
        """
        Get or create Bedrock runtime client.

        Returns:
            boto3.client: Configured Bedrock runtime client
        """
        if self._bedrock_runtime is not None:
            return self._bedrock_runtime

        self._bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.bedrock_region
        )
        return self._bedrock_runtime

    def enhance_query(self, user_query: str) -> str:
        """
        Use LLM to enhance/expand the query for better retrieval.

        Args:
            user_query: Original user question

        Returns:
            Enhanced query string
        """
        try:
            bedrock = self._get_bedrock_runtime_client()

            prompt = f"""Given this user question, generate an improved search query that will help find relevant information.
The improved query should:
- Extract key concepts and terms
- Add relevant synonyms or related terms
- Be concise and focused

User question: {user_query}

Respond with ONLY the improved search query, nothing else."""

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            response = bedrock.invoke_model(
                modelId=self.claude_model_id,
                contentType='application/json',
                accept='application/json',
                body=body
            )

            response_body = json.loads(response['body'].read())
            enhanced_query = response_body['content'][0]['text'].strip()

            print(f"  Original query: {user_query}")
            print(f"  Enhanced query: {enhanced_query}")

            return enhanced_query

        except Exception as e:
            print(f"Warning: Query enhancement failed: {e}")
            print(f"  Using original query: {user_query}")
            return user_query

    def select_search_strategy(self, query: str) -> str:
        """
        Automatically select the best search strategy based on query analysis.

        Args:
            query: User query

        Returns:
            Strategy name: 'text', 'vector', 'hybrid', 'entity', or 'combined'
        """
        query_lower = query.lower()

        # Check for entity-specific queries
        entity_keywords = ['who', 'person', 'people', 'company', 'companies', 'organization']
        if any(keyword in query_lower for keyword in entity_keywords):
            return 'entity'

        # Check for keyword-heavy queries (better for text search)
        keyword_indicators = ['exactly', 'specifically', 'called', 'named', 'term']
        if any(indicator in query_lower for indicator in keyword_indicators):
            return 'text'

        # Check for semantic/conceptual queries (better for vector search)
        semantic_indicators = ['explain', 'understand', 'concept', 'meaning', 'similar', 'like']
        if any(indicator in query_lower for indicator in semantic_indicators):
            return 'vector'

        # Default to hybrid for balanced results
        return 'hybrid'

    def retrieve_context(
        self,
        query: str,
        num_chunks: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks from OpenSearch.

        Args:
            query: Search query
            num_chunks: Number of chunks to retrieve (uses default if None)
            strategy: Search strategy (auto-selects if None)

        Returns:
            List of relevant chunk dictionaries
        """
        num_chunks = num_chunks or self.max_chunks

        # Auto-select strategy if not specified
        if strategy is None:
            strategy = self.select_search_strategy(query)
            print(f"  Selected strategy: {strategy}")

        # Execute search based on strategy
        try:
            if strategy == 'text':
                chunks = self.indexer.search_text(query, size=num_chunks)
            elif strategy == 'vector':
                chunks = self.indexer.search_vector(query, size=num_chunks)
            elif strategy == 'entity':
                # Try entity search, fall back to hybrid if no results
                chunks = self.indexer.search_by_entity(entity_value=query, size=num_chunks)
                if not chunks:
                    print(f"  No entity results, falling back to hybrid search")
                    chunks = self.indexer.search_hybrid(query, size=num_chunks)
            elif strategy == 'combined':
                # Combined search with entity filtering
                chunks = self.indexer.search_combined(text_query=query, size=num_chunks)
            else:  # hybrid
                chunks = self.indexer.search_hybrid(
                    query,
                    size=num_chunks,
                    text_weight=0.5,
                    vector_weight=0.5
                )

            print(f"  Retrieved {len(chunks)} chunks")
            return chunks

        except Exception as e:
            print(f"Warning: Search failed: {e}")
            return []

    def _format_sources(self, chunks: List[Dict]) -> str:
        """
        Format chunk sources for citation.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted source citations string
        """
        if not chunks:
            return "No sources available."

        sources = []
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get('chunk_id', 'unknown')
            score = chunk.get('score', 0)
            preview = chunk.get('content', '')[:100].replace('\n', ' ')

            source = f"[{i}] Chunk {chunk_id} (score: {score:.3f})\n    {preview}..."

            # Add entities if available
            entities = chunk.get('entities', [])
            if entities:
                entity_summary = ', '.join([f"{e['value']} ({e['type']})" for e in entities[:3]])
                source += f"\n    Entities: {entity_summary}"

            # Add topics if available
            topics = chunk.get('topics', [])
            if topics:
                source += f"\n    Topics: {', '.join(topics[:3])}"

            sources.append(source)

        return "\n\n".join(sources)

    def _build_prompt(
        self,
        query: str,
        context: List[Dict],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the RAG prompt for Claude.

        Args:
            query: User question
            context: Retrieved chunks
            history: Conversation history

        Returns:
            Formatted prompt string
        """
        # Format context
        context_text = "\n\n".join([
            f"[Chunk {chunk.get('chunk_id', i)}]\n{chunk.get('content', '')}"
            for i, chunk in enumerate(context, 1)
        ])

        # Build base prompt
        prompt = f"""You are a helpful assistant answering questions based on provided context.

CONTEXT FROM DOCUMENTS:
{context_text}

"""

        # Add conversation history if available
        if history:
            prompt += "CONVERSATION HISTORY:\n"
            for turn in history[-5:]:  # Last 5 turns
                prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        # Add current question
        prompt += f"""USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question using ONLY information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific chunks when making claims (e.g., "According to Chunk 1...")
4. Be concise but thorough
5. If there's conflicting information in the context, acknowledge it

ANSWER:"""

        return prompt

    def generate_response(
        self,
        query: str,
        context: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate answer using Claude with retrieved context.

        Args:
            query: User question
            context: Retrieved chunks for context
            conversation_history: Previous conversation turns

        Returns:
            Generated answer
        """
        try:
            bedrock = self._get_bedrock_runtime_client()

            # Build prompt
            prompt = self._build_prompt(query, context, conversation_history)

            # Call Claude
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            response = bedrock.invoke_model(
                modelId=self.claude_model_id,
                contentType='application/json',
                accept='application/json',
                body=body
            )

            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text'].strip()

            return answer

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"\n{error_msg}")
            return error_msg

    def ask(
        self,
        question: str,
        num_chunks: Optional[int] = None,
        enhance_query: bool = True,
        include_sources: bool = True
    ) -> Dict[str, any]:
        """
        Main RAG method: retrieve context and generate answer.

        Args:
            question: User question
            num_chunks: Number of chunks to retrieve
            enhance_query: Whether to enhance query with LLM
            include_sources: Whether to include source citations

        Returns:
            Dictionary with 'answer', 'sources', 'chunks', 'enhanced_query'
        """
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}\n")

        # Step 1: Enhance query
        if enhance_query:
            print("Step 1: Enhancing query...")
            enhanced_query = self.enhance_query(question)
            search_query = enhanced_query
        else:
            search_query = question
            enhanced_query = question

        # Step 2: Retrieve context
        print("\nStep 2: Retrieving relevant chunks...")
        chunks = self.retrieve_context(search_query, num_chunks)

        if not chunks:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': "No sources found.",
                'chunks': [],
                'enhanced_query': enhanced_query
            }

        # Step 3: Generate answer
        print("\nStep 3: Generating answer with Claude...")
        answer = self.generate_response(question, chunks, self.conversation_history)

        # Step 4: Format sources
        sources = self._format_sources(chunks) if include_sources else ""

        print(f"\n{'='*70}")
        print("Answer generated successfully!")
        print(f"{'='*70}\n")

        return {
            'answer': answer,
            'sources': sources,
            'chunks': chunks,
            'enhanced_query': enhanced_query
        }

    def chat(self, message: str, num_chunks: Optional[int] = None) -> Dict[str, any]:
        """
        Multi-turn conversation with history management.

        Args:
            message: User message
            num_chunks: Number of chunks to retrieve

        Returns:
            Dictionary with response and metadata
        """
        # Get response
        result = self.ask(message, num_chunks=num_chunks, enhance_query=True)

        # Update conversation history
        self.conversation_history.append({
            'user': message,
            'assistant': result['answer']
        })

        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return result

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.

        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy()

    def get_stats(self) -> Dict:
        """
        Get RAG system statistics.

        Returns:
            Dictionary with configuration and stats
        """
        return {
            'claude_model': self.claude_model_id,
            'max_chunks': self.max_chunks,
            'temperature': self.temperature,
            'max_history': self.max_history,
            'current_history_length': len(self.conversation_history),
            'index_name': self.indexer.opensearch_index_name,
            'opensearch_host': self.indexer.opensearch_host
        }
