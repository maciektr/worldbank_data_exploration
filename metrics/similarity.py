import os
import platform
import pandas as pd
import multiprocessing as mp

from metrics.rbind import tsdist
from data_sources.preprocessing import preprocess_df
from data_sources.load_dataset import load_dataset, split_by_columns
import settings


TSDIST_METRICS = [
    "DTWDistance",
    "STSDistance",
    "DissimDistance",
    "CorDistance",
    "CCorDistance",
    "FourierDistance",
]


def write_file(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def get_metric_path(cache_dir, indicator, metric):
    return os.path.join(cache_dir, indicator, "tsdist", f"{metric}.csv")


def frame_processor(params):
    indicator, frame, cache_dir, metrics = params
    frame = preprocess_df(frame)
    for metric in metrics:
        corr = frame.corr(method=tsdist(metric))
        path = get_metric_path(cache_dir, indicator, metric)
        write_file(path, corr)


class SimilarityBuilder:
    def __init__(self, base_dir=settings.CACHE_DIR, metrics=None):
        self.cache_dir = os.path.join(
            os.path.abspath(base_dir),
            "similarity",
        )

        if not metrics:
            metrics = TSDIST_METRICS
        self.metrics = metrics

    def run(self, frames, parallel=True):
        params = [
            (ind, frame, self.cache_dir, self.metrics) for ind, frame in frames.items()
        ]
        if parallel:
            with mp.Pool(settings.PROCESSING_POOL) as p:
                p.map(frame_processor, params)
        else:
            for p in params:
                frame_processor(p)

    def load(self, indicators):
        result = {}
        for indicator in indicators:
            result[indicator] = {}
            for metric in self.metrics:
                path = get_metric_path(self.cache_dir, indicator, metric)
                result[indicator][metric] = pd.read_csv(path)
        return result


if __name__ == "__main__":
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True)

    sim = SimilarityBuilder()
    df = load_dataset()
    frames = split_by_columns(df)
    sim.run(frames)
