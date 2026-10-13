"""Automated Dataset Downloader and Manifest for Benchmark Datasets.

Provides automated download instructions, persistent URLs, and metadata loaders for:
1. SKIPP'D (Stanford Sky Images and Photovoltaic Power Dataset)
2. Folsom Dataset (UC Davis, Pedro et al. 2019)
3. NREL SRRL BMS (National Renewable Energy Lab Baseline Measurement System)
"""

import os
import json
import urllib.request
from typing import Dict, Any


DATASET_CATALOG = {
    "skippd": {
        "name": "SKIPP'D (Stanford SKy Images and Photovoltaic Power Dataset)",
        "source": "Stanford Environmental Assessment and Optimization (EAO) Group",
        "description": "3 years (2017-2019) of 1-min 640x480 all-sky images paired with 1-min GHI and 30kW PV power generation.",
        "purl_url": "https://purl.stanford.edu/dj417rh1007",
        "huggingface_repo": "solarbench/SKIPPD",
        "citation": "Nie, Y., et al. (2023). SKIPP'D: A SKy Images and Photovoltaic Power Generation Dataset. Solar Energy.",
        "download_instructions": [
            "Option A (Hugging Face Datasets):",
            "  pip install datasets",
            "  from datasets import load_dataset",
            "  dataset = load_dataset('solarbench/SKIPPD')",
            "",
            "Option B (Stanford Digital Repository direct download):",
            "  Visit https://purl.stanford.edu/dj417rh1007 and download 2017_2019_benchmark.zip"
        ]
    },
    "folsom": {
        "name": "Folsom Solar Irradiance and Sky Images Dataset",
        "source": "UC Davis (Pedro, Larson, & Coimbra, 2019)",
        "description": "3 years (2014-2016) of 1-second GHI and DNI ground radiometers with 1-min all-sky camera images in Folsom, CA.",
        "zenodo_doi": "10.5281/zenodo.2826939",
        "zenodo_url": "https://zenodo.org/record/2826939",
        "citation": "Pedro, H. T., Larson, D. P., & Coimbra, C. F. (2019). A comprehensive dataset for solar forecasting. Renewable Energy.",
        "download_instructions": [
            "Download Zenodo archive using Zenodo API or browser:",
            "  https://zenodo.org/record/2826939/files/folsom_ground_data.zip",
            "  https://zenodo.org/record/2826939/files/folsom_sky_images_2015.zip"
        ]
    },
    "nrel_srrl": {
        "name": "NREL SRRL (Solar Radiation Research Laboratory - BMS & TSI880)",
        "source": "National Renewable Energy Laboratory, Golden, CO",
        "description": "Gold standard research radiometer baseline (GHI, DNI, DHI, Linke Turbidity) paired with Total Sky Imager TSI-880 images.",
        "web_portal": "https://midc.nrel.gov/srrl_bms/",
        "api_endpoint": "https://developer.nrel.gov/api/solar/data_query/v1",
        "citation": "NREL Solar Radiation Research Laboratory Baseline Measurement System (BMS). Golden, Colorado.",
        "download_instructions": [
            "Access NREL MIDC direct export portal at https://midc.nrel.gov/srrl_bms/",
            "Select parameters: Global Horizontal (CMP22 / PSP), Direct Normal (NIP), Sky Imager (TSI-880)."
        ]
    }
}


class DatasetDownloader:
    """Manages benchmark dataset discovery and directory manifests."""

    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        os.makedirs(data_root, exist_ok=True)

    def print_catalog(self):
        """Displays available benchmark datasets and how to download them."""
        print("=" * 80)
        print("SOLAR IRRADIANCE & SKY IMAGE BENCHMARK DATASET CATALOG")
        print("=" * 80)
        for key, item in DATASET_CATALOG.items():
            print(f"\n[{key.upper()}] - {item['name']}")
            print(f"  Source: {item['source']}")
            print(f"  Description: {item['description']}")
            print("  Instructions:")
            for line in item["download_instructions"]:
                print(f"    {line}")
        print("=" * 80)

    def export_catalog_json(self, target_path: str = "data/dataset_catalog.json"):
        """Exports the catalog metadata to a JSON file."""
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(DATASET_CATALOG, f, indent=2)
        print(f"Exported dataset catalog manifest to {target_path}")
