import os

PROCESSING_POOL = 15

DATA_DIR = os.path.join(os.getenv("PWD", os.getcwd()), "data/")
CACHE_DIR = os.path.join(os.getenv("PWD", os.getcwd()), "cache/")
