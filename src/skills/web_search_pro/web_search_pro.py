import requests
from bs4 import BeautifulSoup
import urllib.parse
from langchain_core.tools import tool

@tool
def search_web(query: str, search_type: str = 'general'):
    """
    Performs a web search for news, images, or videos using public search engines.

    Args:
        query: The search term.
        search_type: Type of search: 'general', 'news', 'images', or 'videos'.

    Returns:
        A list of dictionaries containing 'title', 'link', and 'snippet'.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    base_url = "https://html.duckduckgo.com/html/"

    search_query = query
    if search_type == 'news':
        search_query += " news"
    elif search_type == 'images':
        search_query += " images"
    elif search_type == 'videos':
        search_query += " videos"

    try:
        response = requests.post(base_url, data={'q': search_query}, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result'):
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')

            if title_tag:
                link = title_tag['href']
                if 'uddg=' in link:
                    link = urllib.parse.unquote(link.split('uddg=')[1].split('&')[0])

                results.append({
                    'title': title_tag.get_text(strip=True),
                    'link': link,
                    'snippet': snippet_tag.get_text(strip=True) if snippet_tag else ""
                })

        return results[:10]
    except Exception as e:
        return f"Error during search: {str(e)}"
