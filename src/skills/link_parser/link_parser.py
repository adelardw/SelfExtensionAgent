import urllib.request
from bs4 import BeautifulSoup
import re
import json
from langchain_core.tools import tool

@tool
def parse_link(url: str, query: str = None, chunk_size: int = 2000) -> str:
    """
    Parses a given URL to extract relevant information, optionally focusing on a specific query.
    Handles large contexts by chunking and applying hybrid search methods.

    Args:
        url: The URL to parse.
        query: Optional query to focus the extraction (e.g., "summarize" or "find contact details").
        chunk_size: Size of text chunks for processing large contexts (default: 2000 characters).

    Returns:
        str: Extracted and summarized information from the link, or an error message.
    """
    try:
        # Fetch the webpage content
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode('utf-8')

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()

        # Get clean text content
        text = soup.get_text(separator='\n', strip=True)

        # Split into manageable chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Process query if provided
        if query:
            query = query.lower()
            if "summarize" in query:
                # Placeholder for actual summarization logic
                summary = chunks[0][:500] + "..." if len(chunks[0]) > 500 else chunks[0]
                return f"Summary: {summary}"
            elif "contact" in query:
                # Extract contact details
                emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
                phones = re.findall(r'\+?\d[\d\s-]{7,}\d', text)
                return json.dumps({"emails": emails, "phones": phones}, indent=2)
            else:
                # Semantic search placeholder (replace with actual implementation)
                return f"Relevant content based on '{query}': {chunks[0]}"
        else:
            # Return the first chunk if no query is provided
            return f"Content: {chunks[0]}"

    except urllib.error.URLError as e:
        return f"Failed to fetch the URL: {e.reason}"
    except Exception as e:
        return f"Error parsing the link: {str(e)}"