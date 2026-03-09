import json
import urllib.request
import urllib.parse
from langchain_core.tools import tool

@tool
def weather_check(location: str) -> str:
    """Fetch current weather conditions for a specified location."""
    try:
        encoded_location = urllib.parse.quote(location)
        url = f'https://wttr.in/{encoded_location}?format=%C+%t+%w+%h'
        req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.68.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode().strip()
        return f"Current weather in {location}: {data}"
    except Exception as e:
        return f"Error fetching weather data: {e}"