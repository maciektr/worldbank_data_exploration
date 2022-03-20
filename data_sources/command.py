import os
import shutil

import settings
from command import BaseCommand


class Command(BaseCommand):
    def __init__(self, *args, **kwargs):
        super(Command, self).__init__(*args, **kwargs)

    def add_arguments(self, parser):
        parser.add_argument(
            "-c", "--clear", type=str, nargs="?", default=None, help="clear data"
        )

    @staticmethod
    def remove(cmd):
        if cmd == "all":
            files = os.listdir(settings.DATA_DIR)
            files = filter(lambda x: x != ".gitignore", files)
            files = map(lambda x: os.path.join(settings.DATA_DIR, x), files)
            for file in files:
                shutil.rmtree(file)
        if cmd == "cache":
            path = os.path.join(settings.DATA_DIR, "cache/")
            if os.path.exists(path):
                shutil.rmtree(path)
        return 0

    def handle(self, *args, **kwargs):
        if kwargs["clear"]:
            return self.__class__.remove(kwargs["clear"])
