import json
import urllib.request
from langchain_core.tools import tool

@tool
def get_orcid_works(orcid_id: str) -> str:
    """Fetch all works/publications from an ORCID profile via the public API.

    Args:
        orcid_id: ORCID identifier (e.g. '0000-0003-0396-0333') with or without hyphens.

    Returns:
        JSON string with list of works including titles, publication years, DOIs, etc.
    """
    try:
        # Normalize ORCID ID
        clean_id = orcid_id.strip().replace('-', '')
        formatted_id = f'{clean_id[:4]}-{clean_id[4:8]}-{clean_id[8:12]}-{clean_id[12:]}'
        
        url = f'https://pub.orcid.org/v3.0/{formatted_id}/works'
        req = urllib.request.Request(
            url,
            headers={
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0'
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        works = data.get('group', [])
        result = []
        for group in works:
            for summary in group.get('work-summary', []):
                title = summary.get('title', {}).get('title', {}).get('value', '')
                pub_date = summary.get('publication-date', {})
                year = None
                if pub_date and pub_date.get('year', {}).get('value'):
                    year = pub_date['year']['value']
                path = summary.get('path', '')
                doi = ''
                for ext_id in summary.get('external-ids', {}).get('external-id', []):
                    if ext_id.get('external-id-type') == 'doi':
                        doi = ext_id.get('external-id-value', '')
                result.append({
                    'title': title,
                    'year': year,
                    'doi': doi,
                    'path': path
                })
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({'error': f'Failed to fetch ORCID works: {str(e)}'})

@tool
def count_pre_2020_works(works_json: str) -> str:
    """Count works published before 2020 from the JSON output of get_orcid_works.

    Args:
        works_json: The JSON string returned by get_orcid_works.

    Returns:
        Number of works with publication year before 2020.
    """
    try:
        works = json.loads(works_json)
        if isinstance(works, dict) and 'error' in works:
            return f'Error: {works["error"]}'
        
        count = 0
        years = []
        for w in works:
            y = w.get('year')
            if y and y.isdigit():
                year_int = int(y)
                years.append(year_int)
                if year_int < 2020:
                    count += 1
        return json.dumps({
            'total_works': len(works),
            'pre_2020_count': count,
            'years': years
        }, ensure_ascii=False)
    except Exception as e:
        return f'Error counting works: {str(e)}'