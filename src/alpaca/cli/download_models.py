#!/usr/bin/env python3
"""Download ALPaCA models from Zenodo."""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlopen
import shutil
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

from ..logger import log, console


ZENODO_URL = "https://zenodo.org/records/17215591/files/ALPaCA%20v1.0.0.zip?download=1"
MODEL_PATH_IN_ZIP = "ALPaCA 2/inst/extdata/"


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
    log.info(f"Downloading ALPaCA models from Zenodo (2.3 GB)...")
    log.info(f"URL: {ZENODO_URL}")

    try:
        with urlopen(ZENODO_URL) as response:
            total_size = int(response.info().get('Content-Length', 0))
            
            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
                console=console # Use the shared console
            ) as progress:
                task = progress.add_task("Downloading", total=total_size, filename="ALPaCA models")
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(16 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    except Exception as e:
        log.error(f"Error downloading: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

    # Extract .pt files
    log.info("Extracting model files...")
    pt_files_extracted = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            pt_files = [f for f in zf.namelist()
                       if f.startswith(MODEL_PATH_IN_ZIP) and f.endswith('.pt')]

            if not pt_files:
                log.error("Error: No .pt files found in archive")
                return False

            for file in pt_files:
                filename = Path(file).name
                log.info(f"  - {filename}")
                data = zf.read(file)
                (output_dir / filename).write_bytes(data)
                pt_files_extracted.append(filename)
    except Exception as e:
        log.error(f"Error extracting: {e}")
        return False
    finally:
        if zip_path.exists():
            zip_path.unlink()

    log.info(f"\n[green]✓[/green] Downloaded {len(pt_files_extracted)} model files")
    log.info(f"[green]✓[/green] Location: {output_dir.absolute()}")
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
