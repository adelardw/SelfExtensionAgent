import json
import urllib.request
import urllib.parse
from langchain_core.tools import tool

@tool
def pubchem_search_by_classification(name: str, max_results: int = 500) -> str:
    """Search PubChem compounds by a classification name (e.g. 'Food Additive Status')."""
    try:
        encoded = urllib.parse.quote(name)
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/classification/name/{encoded}/cids/JSON'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
            cids = data['IdentifierList']['CID'][:max_results]
            return json.dumps({'cids': cids, 'count': len(cids)})
        return json.dumps({'error': 'No results found for classification: ' + name})
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        return json.dumps({'error': f'HTTP {e.code}: {body[:200]}'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@tool
def pubchem_get_compound_properties(cids: str, properties: str = 'MolecularWeight,HeavyAtomCount,HBondAcceptorCount,Complexity,MolecularFormula') -> str:
    """Get properties for specific PubChem compound CIDs."""
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/{properties}/JSON'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            props = data['PropertyTable']['Properties']
            return json.dumps({'properties': props}, indent=2)
        return json.dumps({'error': 'Could not retrieve properties'})
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        return json.dumps({'error': f'HTTP {e.code}: {body[:300]}'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@tool
def pubchem_get_compound_enzymes(cid: str) -> str:
    """Get enzyme transformations for a PubChem compound."""
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/enzymes/JSON'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return json.dumps(data, indent=2)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        return json.dumps({'error': f'HTTP {e.code}: {body[:300]}'})
    except Exception as e:
        return json.dumps({'error': str(e)})

@tool
def pubchem_get_gene_cooccurrences(cid: str) -> str:
    """Get gene-chemical co-occurrences for a PubChem compound."""
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/geneid/JSON'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return json.dumps(data, indent=2)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        return json.dumps({'error': f'HTTP {e.code}: {body[:300]}'})
    except Exception as e:
        return json.dumps({'error': str(e)})