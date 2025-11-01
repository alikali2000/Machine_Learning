"""
Main script to demonstrate semantic chunking
"""

from semantic_chunker import SemanticChunker


def main():
    """Example usage of SemanticChunker"""
    # DEBUG: Uncomment the line below to add a breakpoint here
    # breakpoint()

    # Initialize chunker with custom parameters
    chunker = SemanticChunker(
        chunk_size=1000,      # Target 1000 characters per chunk
        chunk_overlap=200,    # 200 character overlap between chunks
        min_chunk_size=100    # Minimum 100 characters per chunk
    )

    # DEBUG: Inspect chunker configuration
    print(f"[DEBUG] Chunker initialized with:")
    print(f"  - chunk_size: {chunker.chunk_size}")
    print(f"  - chunk_overlap: {chunker.chunk_overlap}")
    print(f"  - min_chunk_size: {chunker.min_chunk_size}\n")

    # Chunk the notebook file
    input_file = "notebooks/Decision Trees and Random Forest Project.ipynb"
    print(f"Reading and chunking: {input_file}")

    # DEBUG: Uncomment to pause before chunking
    # breakpoint()

    chunks = chunker.chunk_file(input_file)

    # DEBUG: Show chunk count
    print(f"[DEBUG] Generated {len(chunks)} chunks\n")

    # Print summary
    chunker.print_chunk_summary(chunks)

    # Save chunks to JSON
    output_file = "chunks_output.json"
    chunker.save_chunks(chunks, output_file)

    print(f"\nAll chunks have been saved to '{output_file}'")

    # ========================================
    # S3 EXAMPLE (Uncomment to use)
    # ========================================
    # To use S3, first install boto3: pip install boto3
    #
    # Option 1: Use default AWS credentials (from ~/.aws/credentials or environment variables)
    # s3_chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
    # s3_file = "s3://my-bucket/path/to/document.txt"
    # s3_chunks = s3_chunker.chunk_file(s3_file)
    # s3_chunker.save_chunks(s3_chunks, "s3_chunks_output.json")
    #
    # Option 2: Pass AWS credentials explicitly
    # s3_chunker = SemanticChunker(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     aws_access_key_id="YOUR_ACCESS_KEY",
    #     aws_secret_access_key="YOUR_SECRET_KEY",
    #     aws_region="us-east-1"
    # )
    # s3_chunks = s3_chunker.chunk_file("s3://my-bucket/data.ipynb")
    # print(f"S3: Generated {len(s3_chunks)} chunks")


if __name__ == "__main__":
    main()
