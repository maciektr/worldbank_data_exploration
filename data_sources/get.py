from data_sources.build_package import build_packages
from pandas_datapackage_reader import read_datapackage
from typing import List
import pandas


def get_indicators(indicators: List[str], **kwargs):
    paths = build_packages(indicators, **kwargs)
    dataframes = [read_datapackage(path) for path in paths]
    if len(dataframes) > 0:
        return pandas.concat(dataframes)
    return dataframes[0]
