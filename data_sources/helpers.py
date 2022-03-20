from urllib.parse import urlencode, urljoin
from typing import Dict, Optional
from functools import reduce


def build_url(*path, query: Optional[Dict[str, str]] = None):
    path = reduce(urljoin, path)
    if query:
        path += "?" + urlencode(query)
    return path
