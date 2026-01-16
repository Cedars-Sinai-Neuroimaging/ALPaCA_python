#!/usr/bin/env python3
"""Download ALPaCA models from Zenodo."""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/17215591/files/ALPaCA%20v1.0.0.zip?download=1"
MODEL_PATH_IN_ZIP = "ALPaCA 2/inst/extdata/"


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_models(output_dir=None):
    """Download and extract ALPaCA model weights from Zenodo."""

    if output_dir is None:
        # Default: models/ in package root
        output_dir = Path(__file__).parent.parent / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "alpaca_temp.zip"

    # Download
    print(f"Downloading ALPaCA models from Zenodo (2.3 GB)...")
    print(f"URL: {ZENODO_URL}")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Download') as t:
            urlretrieve(ZENODO_URL, zip_path, reporthook=t.update_to)
    except Exception as e:
        print(f"Error downloading: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

    # Extract .pt files
    print("Extracting model files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            pt_files = [f for f in zf.namelist()
                       if f.startswith(MODEL_PATH_IN_ZIP) and f.endswith('.pt')]

            if not pt_files:
                print("Error: No .pt files found in archive")
                return False

            for file in pt_files:
                filename = Path(file).name
                print(f"  {filename}")
                data = zf.read(file)
                (output_dir / filename).write_bytes(data)
    except Exception as e:
        print(f"Error extracting: {e}")
        return False
    finally:
        if zip_path.exists():
            zip_path.unlink()

    print(f"\n✓ Downloaded {len(pt_files)} model files")
    print(f"✓ Location: {output_dir.absolute()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download ALPaCA model weights from Zenodo',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Output directory for models (default: <package>/models/)'
    )

    args = parser.parse_args()

    success = download_models(args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
