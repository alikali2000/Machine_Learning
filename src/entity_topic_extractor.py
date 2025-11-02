"""
Entity and Topic Extraction Module using Claude Sonnet API

This module provides functionality to extract entities (people, organizations,
locations, dates) and topics from text chunks using Anthropic's Claude Sonnet model.
"""

import json
import os
from typing import Dict, List, Optional
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EntityTopicExtractor:
    """Extracts entities and topics from text using Claude Sonnet API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor with API credentials.

        Args:
            api_key: Anthropic API key. If not provided, will attempt to load from
                    ANTHROPIC_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Please set ANTHROPIC_API_KEY "
                "environment variable or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"  # Latest Claude Sonnet model

    def extract(self, text: str) -> Dict[str, List]:
        """
        Extract entities and topics from the given text.

        Args:
            text: The text content to analyze

        Returns:
            Dictionary containing:
                - entities: List of extracted entities with type and value
                - topics: List of main topics/themes identified in the text

        Example:
            {
                "entities": [
                    {"type": "PERSON", "value": "John Smith"},
                    {"type": "ORG", "value": "OpenAI"},
                    {"type": "DATE", "value": "2024"}
                ],
                "topics": ["machine learning", "artificial intelligence"]
            }
        """
        try:
            prompt = self._build_extraction_prompt(text)

            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response text
            response_text = message.content[0].text

            # Parse JSON response
            result = json.loads(response_text)

            # Validate structure
            if "entities" not in result or "topics" not in result:
                raise ValueError("Invalid response structure from API")

            return result

        except json.JSONDecodeError as e:
            print(f"Error parsing API response as JSON: {e}")
            return {"entities": [], "topics": []}
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return {"entities": [], "topics": []}

    def _build_extraction_prompt(self, text: str) -> str:
        """Build the prompt for entity and topic extraction."""
        return f"""Analyze the following text and extract:
1. Entities: People, organizations, locations, and dates
2. Topics: Main themes or subjects discussed

Text to analyze:
{text}

Respond ONLY with valid JSON in this exact format:
{{
  "entities": [
    {{"type": "PERSON", "value": "name"}},
    {{"type": "ORG", "value": "organization"}},
    {{"type": "LOC", "value": "location"}},
    {{"type": "DATE", "value": "date"}}
  ],
  "topics": ["topic1", "topic2", "topic3"]
}}

Entity types:
- PERSON: People's names
- ORG: Organizations, companies, institutions
- LOC: Locations, places, geographical entities
- DATE: Dates, times, temporal expressions

Keep topics concise (2-5 words each). Extract only the most relevant topics (typically 2-5 topics per text).
"""


def extract_entities_topics(text: str, api_key: Optional[str] = None) -> Dict[str, List]:
    """
    Convenience function to extract entities and topics from text.

    Args:
        text: The text content to analyze
        api_key: Optional Anthropic API key

    Returns:
        Dictionary with 'entities' and 'topics' lists
    """
    extractor = EntityTopicExtractor(api_key=api_key)
    return extractor.extract(text)
