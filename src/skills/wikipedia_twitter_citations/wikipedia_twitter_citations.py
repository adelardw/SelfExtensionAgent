import json
import re
import urllib.parse
import urllib.request
from langchain_core.tools import tool


def _api_call(params: dict) -> dict:
    """Make a Wikipedia API call and return parsed JSON."""
    base = "https://en.wikipedia.org/w/api.php"
    params["format"] = "json"
    params["origin"] = "*"
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "WikipediaTwitterSkill/1.0 (research bot)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def _last_june2023_revision(title: str) -> int:
    """
    Get the last revision ID of a Wikipedia page from June 2023.
    We search for revisions older than (rvend) 2023-07-01T00:00:00Z,
    ordered by rvdir=older (newest first), limit=1 → gets the newest
    revision that is still before July 2023.
    """
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "1",
        "rvdir": "older",
        "rvstart": "2023-07-01T00:00:00Z",
        "rvprop": "ids|timestamp",
    }
    data = _api_call(params)
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            raise ValueError(f"Page '{title}' does not exist.")
        revisions = page_data.get("revisions", [])
        if not revisions:
            raise ValueError(f"No revision found for '{title}' before July 2023.")
        return revisions[0]["revid"]
    raise ValueError(f"Could not find page '{title}'.")


def _count_twitter_links_in_revision(title: str, revid: int) -> int:
    """
    Fetch the parsed HTML content of a Wikipedia page at a specific revision
    and count external links to twitter.com or x.com.
    """
    params = {
        "action": "parse",
        "page": title,
        "oldid": str(revid),
        "prop": "text",
    }
    data = _api_call(params)
    html = data.get("parse", {}).get("text", {}).get("*", "")
    if not html:
        return 0

    # Find all external links in the HTML. Wikipedia wraps external links in
    # <a class="external text" href="..."> or <a rel="mw:ExtLink" href="...">
    # We look for href attributes containing twitter.com or x.com
    twitter_count = 0
    for match in re.finditer(r'href\s*=\s*"([^"]*)"', html, re.IGNORECASE):
        href = match.group(1).lower()
        if "twitter.com" in href or "x.com" in href:
            # Avoid counting Wikipedia internal URLs that might have 'x.com' as subdomain
            if href.startswith("http://") or href.startswith("https://"):
                twitter_count += 1
    return twitter_count


@tool
def count_twitter_citations_per_day() -> str:
    """
    Count Twitter/X post citations on English Wikipedia pages for each day of August
    (August 1 through August 31), using the last revision of each page from June 2023.

    Returns a formatted per-day breakdown showing how many Twitter/X references
    were found on each day-of-August Wikipedia page.
    """
    results = {}
    errors = []
    total_all_days = 0

    for day in range(1, 32):
        title = f"August {day}"
        try:
            revid = _last_june2023_revision(title)
            count = _count_twitter_links_in_revision(title, revid)
            results[title] = count
            total_all_days += count
        except Exception as e:
            errors.append(f"{title}: {e}")
            results[title] = -1  # error marker

    lines = []
    lines.append("# Twitter/X Citations on English Wikipedia Date Pages (August 1-31)")
    lines.append("## Based on the last revision from June 2023\n")
    lines.append("| Page | Twitter/X Links Count |")
    lines.append("|------|----------------------|")
    for day in range(1, 32):
        title = f"August {day}"
        count = results.get(title)
        if count is None or count == -1:
            lines.append(f"| {title} | ERROR |")
        else:
            lines.append(f"| {title} | {count} |")

    lines.append(f"\n**Total across all 31 days: {total_all_days}**")

    if errors:
        lines.append(f"\n### Errors encountered:")
        for err in errors:
            lines.append(f"- {err}")

    # Per-day breakdown text
    lines.append("\n\n## Per-day breakdown")
    day_counts = {}
    for day in range(1, 32):
        title = f"August {day}"
        c = results.get(title, -1)
        if c >= 0:
            day_counts[str(day)] = c

    if day_counts:
        lines.append(f"Per-day counts: {json.dumps(day_counts, indent=2)}")

    return "\n".join(lines)