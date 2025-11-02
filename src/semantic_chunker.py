"""
Semantic Chunking Script
Chunks text files or documents while preserving semantic boundaries
Supports both local files and AWS S3 objects
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from src.entity_topic_extractor import EntityTopicExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False


class SemanticChunker:
    """
    Chunks text semantically by:
    - Respecting sentence boundaries
    - Grouping related content
    - Maintaining context with overlap
    - Preserving paragraph structure
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        extract_entities_topics: bool = False,
        anthropic_api_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None
    ):
        """
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk
            extract_entities_topics: Whether to extract entities and topics using Claude Sonnet API
            anthropic_api_key: Anthropic API key for entity/topic extraction (optional)
            aws_access_key_id: AWS access key ID (optional, will use env vars or credentials file if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token (optional)
            aws_region: AWS region (optional, defaults to us-east-1)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.extract_entities_topics = extract_entities_topics

        # Entity/Topic Extractor
        self.extractor = None
        if self.extract_entities_topics:
            if not EXTRACTOR_AVAILABLE:
                raise ImportError(
                    "Entity/Topic extraction requires anthropic SDK. "
                    "Install it with: pip install anthropic python-dotenv"
                )
            try:
                self.extractor = EntityTopicExtractor(api_key=anthropic_api_key)
            except ValueError as e:
                print(f"Warning: Could not initialize entity/topic extractor: {e}")
                print("Continuing without entity/topic extraction.")
                self.extract_entities_topics = False

        # AWS credentials
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region = aws_region or 'us-east-1'

        # S3 client cache
        self._s3_client = None

    def _get_s3_client(self):
        """
        Get or create S3 client with credential fallback

        Credentials are tried in this order:
        1. Explicitly provided credentials (constructor params)
        2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        3. AWS credentials file (~/.aws/credentials)
        4. IAM role (if running on AWS infrastructure)

        Returns:
            boto3.client: Configured S3 client

        Raises:
            ImportError: If boto3 is not installed
            NoCredentialsError: If no valid credentials are found
        """
        if not S3_AVAILABLE:
            raise ImportError(
                "boto3 is not installed. Install it with: pip install boto3"
            )

        # Return cached client if available
        if self._s3_client is not None:
            return self._s3_client

        # Build kwargs for boto3 client
        kwargs = {'service_name': 's3', 'region_name': self.aws_region}

        # Use explicitly provided credentials if available
        if self.aws_access_key_id and self.aws_secret_access_key:
            kwargs['aws_access_key_id'] = self.aws_access_key_id
            kwargs['aws_secret_access_key'] = self.aws_secret_access_key
            if self.aws_session_token:
                kwargs['aws_session_token'] = self.aws_session_token

        # Create and cache the client
        # If no explicit credentials, boto3 will automatically try:
        # 1. Environment variables
        # 2. ~/.aws/credentials file
        # 3. IAM role
        self._s3_client = boto3.client(**kwargs)
        return self._s3_client

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """
        Parse S3 URI into bucket and key

        Args:
            s3_uri: S3 URI in format s3://bucket-name/key/path

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URI format is invalid
        """
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with s3://")

        # Remove s3:// prefix
        path = s3_uri[5:]

        # Split into bucket and key
        parts = path.split('/', 1)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid S3 URI: {s3_uri}. Must include bucket and key (s3://bucket/key)"
            )

        bucket = parts[0]
        key = parts[1]

        if not bucket:
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Bucket name cannot be empty")

        if not key:
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Key cannot be empty")

        return bucket, key

    def _read_from_s3(self, s3_uri: str) -> str:
        """
        Read file content from S3

        Args:
            s3_uri: S3 URI (s3://bucket/key)

        Returns:
            File content as string

        Raises:
            ImportError: If boto3 is not installed
            ValueError: If S3 URI is invalid
            ClientError: If S3 operation fails (bucket/key not found, permission denied, etc.)
            NoCredentialsError: If AWS credentials are not configured
        """
        # Parse S3 URI
        bucket, key = self._parse_s3_uri(s3_uri)

        try:
            # Get S3 client
            s3_client = self._get_s3_client()

            # Download object
            response = s3_client.get_object(Bucket=bucket, Key=key)

            # Read and decode content
            content = response['Body'].read().decode('utf-8')

            return content

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')

            if error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {bucket}")
            elif error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: {s3_uri}")
            elif error_code == '403' or error_code == 'AccessDenied':
                raise PermissionError(f"Access denied to S3 object: {s3_uri}")
            else:
                raise RuntimeError(f"S3 error ({error_code}): {str(e)}")

        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. Please configure credentials using:\n"
                "1. Pass credentials to SemanticChunker constructor, or\n"
                "2. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, or\n"
                "3. Configure ~/.aws/credentials file"
            )

    def read_file(self, file_path: str) -> str:
        """
        Read content from local files or S3 objects

        Supports:
        - Local .txt files
        - Local .ipynb (Jupyter notebook) files
        - S3 objects (.txt or .ipynb)

        Args:
            file_path: Local file path or S3 URI (s3://bucket/key)

        Returns:
            File content as string

        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file or S3 object not found
            PermissionError: If access is denied
            RuntimeError: For S3-related errors
        """
        # Detect if S3 URI
        if file_path.startswith('s3://'):
            # Read from S3
            content = self._read_from_s3(file_path)

            # Extract file extension from S3 key for type detection
            path = Path(file_path)
            file_suffix = path.suffix

            # Process based on file type
            if file_suffix == '.txt':
                return content

            elif file_suffix == '.ipynb':
                # Parse JSON content
                notebook = json.loads(content)
                text_parts = []

                for cell in notebook.get('cells', []):
                    cell_type = cell.get('cell_type', '')
                    source = cell.get('source', [])

                    if isinstance(source, list):
                        cell_content = ''.join(source)
                    else:
                        cell_content = source

                    if cell_type == 'markdown':
                        text_parts.append(f"[MARKDOWN]\n{cell_content}\n")
                    elif cell_type == 'code':
                        text_parts.append(f"[CODE]\n{cell_content}\n")

                    # Include outputs for context
                    outputs = cell.get('outputs', [])
                    for output in outputs:
                        if 'text' in output:
                            output_text = output['text']
                            if isinstance(output_text, list):
                                output_text = ''.join(output_text)
                            text_parts.append(f"[OUTPUT]\n{output_text}\n")

                return '\n'.join(text_parts)

            else:
                raise ValueError(f"Unsupported S3 file type: {file_suffix}")

        else:
            # Local file path
            path = Path(file_path)

            if path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif path.suffix == '.ipynb':
                with open(file_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                    text_parts = []

                    for cell in notebook.get('cells', []):
                        cell_type = cell.get('cell_type', '')
                        source = cell.get('source', [])

                        if isinstance(source, list):
                            content = ''.join(source)
                        else:
                            content = source

                        if cell_type == 'markdown':
                            text_parts.append(f"[MARKDOWN]\n{content}\n")
                        elif cell_type == 'code':
                            text_parts.append(f"[CODE]\n{content}\n")

                        # Include outputs for context
                        outputs = cell.get('outputs', [])
                        for output in outputs:
                            if 'text' in output:
                                output_text = output['text']
                                if isinstance(output_text, list):
                                    output_text = ''.join(output_text)
                                text_parts.append(f"[OUTPUT]\n{output_text}\n")

                    return '\n'.join(text_parts)

            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Split on sentence boundaries while preserving the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def create_chunks(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Create semantic chunks from text

        Returns:
            List of dictionaries with chunk content and metadata
        """
        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0

        for para_idx, paragraph in enumerate(paragraphs):
            sentences = self.split_into_sentences(paragraph)

            for sentence in sentences:
                # If adding this sentence exceeds chunk_size, save current chunk
                if len(current_chunk) + len(sentence) > self.chunk_size and len(current_chunk) >= self.min_chunk_size:
                    chunk_dict = {
                        'chunk_id': chunk_id,
                        'content': current_chunk.strip(),
                        'char_count': len(current_chunk.strip()),
                        'paragraph_start': para_idx
                    }

                    # Extract entities and topics if enabled
                    if self.extract_entities_topics and self.extractor:
                        extraction = self.extractor.extract(current_chunk.strip())
                        chunk_dict['entities'] = extraction.get('entities', [])
                        chunk_dict['topics'] = extraction.get('topics', [])

                    chunks.append(chunk_dict)
                    chunk_id += 1

                    # Create overlap by keeping last portion of current chunk
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        # Try to start overlap at sentence boundary
                        overlap_sentences = self.split_into_sentences(overlap_text)
                        if overlap_sentences:
                            current_chunk = ' '.join(overlap_sentences)
                        else:
                            current_chunk = overlap_text
                    else:
                        current_chunk = ""

                current_chunk += " " + sentence if current_chunk else sentence

        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunk_dict = {
                'chunk_id': chunk_id,
                'content': current_chunk.strip(),
                'char_count': len(current_chunk.strip()),
                'paragraph_start': len(paragraphs) - 1
            }

            # Extract entities and topics if enabled
            if self.extract_entities_topics and self.extractor:
                extraction = self.extractor.extract(current_chunk.strip())
                chunk_dict['entities'] = extraction.get('entities', [])
                chunk_dict['topics'] = extraction.get('topics', [])

            chunks.append(chunk_dict)

        return chunks

    def chunk_file(self, file_path: str) -> List[Dict[str, Union[str, int]]]:
        """Read and chunk a file"""
        text = self.read_file(file_path)
        return self.create_chunks(text)

    def save_chunks(self, chunks: List[Dict], output_path: str):
        """Save chunks to a JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_path}")

    def print_chunk_summary(self, chunks: List[Dict]):
        """Print a summary of the chunks"""
        print(f"\n{'='*60}")
        print(f"CHUNKING SUMMARY")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(c['char_count'] for c in chunks) / len(chunks):.0f} characters")
        print(f"Min chunk size: {min(c['char_count'] for c in chunks)} characters")
        print(f"Max chunk size: {max(c['char_count'] for c in chunks)} characters")
        print(f"{'='*60}\n")

        # Show first 3 chunks as preview
        for chunk in chunks[:3]:
            print(f"\n--- Chunk {chunk['chunk_id']} ---")
            print(f"Characters: {chunk['char_count']}")
            preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            print(f"Preview: {preview}\n")
