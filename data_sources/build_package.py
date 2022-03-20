"""
Retrieve indicator data from World Bank Open Data
Convert to datapackage format: https://datahub.io/docs/data-packages
Inspired by: https://github.com/rufuspollock/world-bank-data
"""
from data_sources.helpers import ProgressBar, build_url
from zipfile import ZipFile
import codecs
import csv
import json
import math
import os
import pandas as pd
import settings
import sys
import urllib.parse
import urllib.request


class PackageBuilder:
    BASE_URL = "https://api.worldbank.org/v2/"
    PACKAGES_DIR = "indicators/"

    def __init__(
        self, indicator: str, base_dir: str = settings.DATA_DIR, verbose: bool = False
    ):
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
            query={"downloadformat": "csv"},
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

    def log(self, *message, **kwargs):
        if self.verbose:
            print(*message, **kwargs)

    def execute(self):
        f"""
        Retrieve a world bank indicator and convert to a data package.
        Data Package is stored at {self.data_directory}/indicators/{self.indicator}
        """

        # Download files
        self.retrieve()

        # Process files
        (meta, data) = self.extract()

        datapackage_path = os.path.join(
            self.data_directory, self.__class__.PACKAGES_DIR, meta["name"]
        )
        self.datapackage(meta, data, datapackage_path)
        self.log("Data package written to: ", datapackage_path)

        return datapackage_path

    def retrieve(self):
        cache_dir = os.path.join(self.data_directory, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(self.meta_dest):
            urllib.request.urlretrieve(self.meta_url, self.meta_dest, ProgressBar())

        if not os.path.exists(self.data_dest):
            urllib.request.urlretrieve(self.data_url, self.data_dest, ProgressBar())

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

        data = [fields[:4] + ["Year", "Value"]]
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
    processor = PackageBuilder(indicator_name, base_dir="..", verbose=True)
    processor.execute()
