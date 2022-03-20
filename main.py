import argparse
import sys

from data_sources.command import Command as DataSourcesCommand

parser = argparse.ArgumentParser(description="World Bank Open Data Exploration")
parser.add_argument("command", type=str, help="command to execute")

commands = {"data_sources": DataSourcesCommand(parser)}

if __name__ == "__main__":
    args = parser.parse_args()

    if args.command in commands:
        sys.exit(commands[args.command].handle())
