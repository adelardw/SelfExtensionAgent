import urllib.request
import re
import json
import subprocess
import sys
from langchain_core.tools import tool

# Установка BeautifulSoup
try:
    from bs4 import BeautifulSoup
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "beautifulsoup4"], check=True)
    from bs4 import BeautifulSoup

@tool
def parse_link(url: str, query: str = None, chunk_size: int = 3000) -> str:
    """
    Парсит URL для извлечения информации, опционально фокусируясь на запросе.
    Обрабатывает большие контексты путем разбиения на части.
    """
    try:
        # Fetch the webpage content
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read().decode('utf-8', errors='ignore')

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Get clean text content
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n+', '\n', text) # Убираем лишние пустые строки

        # Split into manageable chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        if not chunks:
            return "Страница пуста или не содержит текста."

        # Process query if provided
        if query:
            query_lower = query.lower()
            if "контакт" in query_lower or "contact" in query_lower:
                emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
                phones = re.findall(r'\+?\d[\d\s-]{7,}\d', text)
                return json.dumps({"emails": list(set(emails)), "phones": list(set(phones))}, indent=2, ensure_ascii=False)
            
            # Поиск наиболее релевантного чанка по ключевым словам
            keywords = query_lower.split()
            best_chunk = chunks[0]
            max_hits = 0
            for chunk in chunks:
                hits = sum(1 for word in keywords if word in chunk.lower())
                if hits > max_hits:
                    max_hits = hits
                    best_chunk = chunk
            return f"Наиболее релевантный фрагмент по запросу '{query}':\n\n{best_chunk}"
        
        # Return the first chunk if no query is provided
        return f"Начало контента страницы:\n\n{chunks[0]}..."

    except Exception as e:
        return f"Ошибка при парсинге ссылки: {str(e)}"
