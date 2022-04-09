"""
Retrieve indicator data from World Bank Open Data
Convert to datapackage format: https://datahub.io/docs/data-packages
Inspired by: https://github.com/rufuspollock/world-bank-data
"""
from data_sources.helpers import build_url
from itertools import repeat
from multiprocessing import Pool
from typing import List
from zipfile import ZipFile
import codecs
import csv
import grequests
import json
import math
import os
import pandas as pd
from data_sources import settings
import sys


class PackageBuilder:
    BASE_URL = "https://api.worldbank.org/v2/"
    PACKAGES_DIR = "indicators/"

    def __init__(
        self,
        indicator: str,
        base_dir: str = settings.DATA_DIR,
        verbose: bool = False,
        overwrite: bool = False,
        download_query=None,
    ):
        if download_query is None:
            download_query = {}
        self.overwrite = overwrite
        self.verbose = verbose

        self.indicator = indicator
        self.log("Processing indicator", self.indicator)

        self.data_directory = os.path.abspath(base_dir)
        self.log("Data directory", self.data_directory)

        self.meta_url = build_url(
            self.__class__.BASE_URL,
            "indicator/",
            self.indicator,
            query={"format": "json"},
        )
        self.data_url = build_url(
            self.__class__.BASE_URL,
            "country/all/indicator/",
            self.indicator,
            query={"downloadformat": "csv", **download_query},
        )
        self.meta_dest = os.path.join(
            self.data_directory, "cache", f"{self.indicator}.meta.json"
        )
        self.data_dest = os.path.join(
            self.data_directory, "cache/", f"{self.indicator}.zip"
        )

        self.log("Meta url: ", self.meta_url)
        self.log("Meta dest: ", self.meta_dest)
        self.log("Data url: ", self.data_url)
        self.log("Data dest: ", self.data_dest)

        self.datapackage_path = os.path.join(
            self.data_directory, self.__class__.PACKAGES_DIR, self.indicator.lower()
        )
        self.log("Datapackage path", self.datapackage_path)

    def log(self, *message, **kwargs):
        if self.verbose:
            print(*message, **kwargs)

    def execute(self):
        f"""
        Retrieve a world bank indicator and convert to a data package.
        Data Package is stored at {self.data_directory}/indicators/{self.indicator}
        """
        # Download files
        responses = grequests.map(self.make_requests())
        self.retrieve(responses)
        return self.datapackage_path

    def retrieve(self, responses):
        # Write cache
        def write_file(resp, path):
            with open(path, "wb") as f:
                f.write(resp.content)

        for response in responses:
            if response.url == self.data_url:
                write_file(response, self.data_dest)
            if response.url == self.meta_url:
                write_file(response, self.meta_dest)

        # Process files
        if os.path.exists(self.datapackage_path) and not self.overwrite:
            return

        (meta, data) = self.extract()

        self.datapackage(meta, data, self.datapackage_path)
        self.log("Data package written to: ", self.datapackage_path)

    def make_requests(self):
        cache_dir = os.path.join(self.data_directory, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        requests_list = []

        if not os.path.exists(self.meta_dest) or self.overwrite:
            requests_list.append(
                grequests.get(
                    self.meta_url,
                )
            )

        if not os.path.exists(self.data_dest) or self.overwrite:
            requests_list.append(grequests.get(self.data_url))

        return requests_list

    def extract(self):
        """
        Extract raw metadata and data into nicely structured form.

        @return: (metadata, data) where metadata is Data Package JSON and data is normalized CSV.
        """
        # Process metadata
        metadata_file = open(self.meta_dest, "r")
        raw_metadata = json.load(metadata_file)[1][0]
        metadata = {
            "title": raw_metadata["name"],
            "name": raw_metadata["id"].lower(),
            "worldbank": {"indicator": raw_metadata["id"].lower()},
            "readme": raw_metadata["sourceNote"],
            "licenses": [{"name": "CC-BY-4.0"}],
            "keywords": [x["value"] for x in raw_metadata["topics"]],
        }

        # Extract data from zip
        zipped_file = ZipFile(self.data_dest, "r")
        name = list(filter(lambda n: n.startswith("API"), zipped_file.namelist()))[0]
        cache_unzipped = codecs.iterdecode(zipped_file.open(name), "utf-8")
        csv_path = self.data_dest + "." + name

        # Preprocess unzipped file
        with open(csv_path, "w") as file:
            # Drop header
            for _ in range(4):
                cache_unzipped.__next__()

            for line in cache_unzipped:
                file.write(line)

        # Read and process data
        df = pd.read_csv(csv_path, header=0)
        df = df.reset_index()

        fields = list(df.columns)
        self.log("Data file fields", fields)

        data = [fields[1:5] + ["Year", "Value"]]
        for index, row in df.iterrows():
            for year, val in zip(fields[5:], row[5:]):
                if not math.isnan(val):
                    data.append(list(row[1:5]) + [year, val])

        return metadata, data

    @classmethod
    def datapackage(self, metadata, data, basepath: str):
        os.makedirs(basepath, exist_ok=True)

        dpjson = os.path.join(basepath, "datapackage.json")
        readme = os.path.join(basepath, "README.md")
        datafp = os.path.join(basepath, "data.csv")

        metadata["resources"] = [
            {
                "name": "data",
                "title": "Indicator data",
                "path": "data.csv",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "utf-8",
                "schema": {
                    "fields": [
                        {
                            "name": "Country Name",
                            "type": "string",
                            "description": "Country or Region name",
                        },
                        {
                            "name": "Country Code",
                            "type": "string",
                            "description": "ISO 3-digit ISO code extended to include regional codes e.g. EUR, ARB etc",
                        },
                        {
                            "name": "Indicator Name",
                            "type": "string",
                            "description": "Indicator name",
                        },
                        {
                            "name": "Indicator Code",
                            "type": "string",
                            "description": "Indicator ID e.g. FP.CPI.TOTL.ZG",
                        },
                        {"name": "Year", "type": "year", "description": "Year"},
                        {
                            "name": "Value",
                            "type": "number",
                            "description": metadata["readme"],
                        },
                    ]
                },
            }
        ]

        with open(dpjson, "w") as fo:
            json.dump(metadata, fo, indent=2)
        with open(readme, "w") as fo:
            fo.write(metadata["readme"])
        with open(datafp, "w") as fo:
            writer = csv.writer(fo)
            writer.writerows(data)


def execute_builders(builders: List[PackageBuilder]):
    requests = []
    for builder in builders:
        requests.extend(builder.make_requests())

    # Download files
    responses = grequests.map(requests)

    for builder in builders:
        builder.retrieve(responses)


def build_package(arg):
    ind, kwargs = arg
    return PackageBuilder(ind, **kwargs).execute()


def build_packages(indicators, concurrent: bool = False, **kwargs):
    # May not work in jupyter notebook due to multiprocessing incompatibility
    if concurrent:
        with Pool(settings.PROCESSING_POOL) as pool:
            return pool.map(build_package, zip(indicators, repeat(kwargs)))

    builders = [PackageBuilder(ind, **kwargs) for ind in indicators]
    execute_builders(builders)
    return [builder.datapackage_path for builder in builders]


if __name__ == "__main__":
    usage = """
Usage: python data_sources/build_package.py {indicator or indicator url}

Example:

    python data_sources/build_package.py GC.DOD.TOTL.GD.ZS
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    indicator_name = sys.argv[1]
    processor = PackageBuilder(indicator_name, verbose=True)
    processor.execute()
