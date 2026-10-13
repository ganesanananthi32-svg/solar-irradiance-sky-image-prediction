"""Dataset discovery and downloader CLI."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.dataset_downloader import DatasetDownloader


def main():
    downloader = DatasetDownloader(data_root="data")
    downloader.print_catalog()
    downloader.export_catalog_json("data/dataset_catalog.json")


if __name__ == "__main__":
    main()
