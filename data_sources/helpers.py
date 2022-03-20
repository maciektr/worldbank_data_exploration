import progressbar
from urllib.parse import urlencode, urljoin
from typing import Dict, Optional
from functools import reduce


def build_url(*path, query: Optional[Dict[str, str]] = None):
    path = reduce(urljoin, path)
    if query:
        path += "?" + urlencode(query)
    return path


class ProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
