#!/usr/bin/env python3
"""
IMD Data Downloader
===================
Downloads publicly accessible IMD / open-climate datasets into data/raw/.

Sources used
------------
1. IRI/LDEO Climate Data Library  (temperature gridded, NCEP reanalysis proxy)
   → Publicly accessible via OpenDAP / direct NetCDF download.

2. IMD Open Data Portal  (imdpune.gov.in)
   → Temperature and rainfall binary grids. Many files require manual
     selection on their web form; this script handles the direct-URL
     accessible files only and prints instructions for the rest.

3. ERA5 via CDS API  (optional, requires cdsapi + free account)
   → Fallback when IMD data is unavailable. Activate with --era5 flag.

Usage
-----
  # Download last 5 years of daily tmax (NCEP via IRI, no auth needed)
  python scripts/download_imd_data.py --product tmax --years 5

  # Download a specific year
  python scripts/download_imd_data.py --product tmax --start-year 2022 --end-year 2023

  # Also pull ERA5 as a fallback
  python scripts/download_imd_data.py --product tmax --years 5 --era5
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_sources import download_file  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# IRI / LDEO  — NCEP reanalysis daily Tmax (0.5° grid, free, no auth)
# URL pattern for daily surface temperature anomaly
# ---------------------------------------------------------------------------
IRI_BASE = (
    "https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1/.DAILY"
    "/.Diagnostic/.above_ground/.maximum/.temp/"
    "T/(1%20Jan%20{year})/(31%20Dec%20{year})RANGEEDGES/"
    "data.nc"
)

# ---------------------------------------------------------------------------
# IMD Open Data  — direct-download binary grids
# Temperature: http://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html
# These direct links are for demonstration; actual IMD bulk files are
# served via their FTP (ftp.imdpune.gov.in) which requires registration.
# ---------------------------------------------------------------------------
IMD_FTP_NOTE = """
╔══════════════════════════════════════════════════════════════════╗
║  IMD Gridded Data — Manual Download Instructions                ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Visit: https://www.imdpune.gov.in/                          ║
║     → Services → Climate Data → Gridded Data Download           ║
║  2. Select product: Maximum Temperature / Minimum Temperature   ║
║  3. Choose year range and click Download                        ║
║  4. Save the .GRD files into:  data/raw/tmax/  or data/raw/tmin/║
║                                                                  ║
║  FTP access (bulk download):                                     ║
║    Host: ftp.imdpune.gov.in                                     ║
║    Register at: https://www.imdpune.gov.in/Clim_Pred_LRF_New/   ║
║    Files: /temperature/MaxTemp/MaxTemp_<YYYY>.GRD               ║
╚══════════════════════════════════════════════════════════════════╝
"""


def download_iri_temperature(start_year: int, end_year: int, out_dir: Path) -> list[Path]:
    """Download NCEP daily Tmax NetCDF files via IRI Data Library (no auth)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for year in range(start_year, end_year + 1):
        url = IRI_BASE.format(year=year)
        dest = out_dir / f"ncep_tmax_{year}.nc"
        if dest.exists():
            log.info("Already exists: %s — skipping", dest.name)
            downloaded.append(dest)
            continue
        log.info("Downloading NCEP Tmax %d …", year)
        try:
            download_file(url, dest)
            log.info("  Saved → %s  (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            downloaded.append(dest)
        except Exception as exc:
            log.error("  Failed for %d: %s", year, exc)
    return downloaded


def download_era5(
    start_year: int,
    end_year: int,
    out_dir: Path,
    variable: str = "2m_temperature",
) -> list[Path]:
    """
    Download ERA5 daily temperature via the Copernicus CDS API.

    Requires:
        pip install cdsapi
        ~/.cdsapirc with your CDS API key
        (Free account: https://cds.climate.copernicus.eu)
    """
    try:
        import cdsapi  # type: ignore
    except ImportError:
        log.error("cdsapi not installed. Run: pip install cdsapi")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    downloaded: list[Path] = []

    for year in range(start_year, end_year + 1):
        dest = out_dir / f"era5_{variable}_{year}.nc"
        if dest.exists():
            log.info("Already exists: %s — skipping", dest.name)
            downloaded.append(dest)
            continue
        log.info("Requesting ERA5 %s %d via CDS API …", variable, year)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": variable,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": "12:00",
                "area": [38, 66, 6, 100],  # North, West, South, East (India bbox)
                "format": "netcdf",
            },
            str(dest),
        )
        log.info("  Saved → %s  (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        downloaded.append(dest)

    return downloaded


def print_imd_instructions() -> None:
    print(IMD_FTP_NOTE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IMD / reanalysis weather data for the ML pipeline"
    )
    parser.add_argument(
        "--product",
        choices=["tmax", "tmin", "rain"],
        default="tmax",
        help="Weather variable to download (default: tmax)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of recent years to download (default: 5)",
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--era5",
        action="store_true",
        help="Also download ERA5 data via cdsapi (requires ~/.cdsapirc)",
    )
    parser.add_argument(
        "--imd-instructions",
        action="store_true",
        help="Print manual IMD download instructions and exit",
    )
    args = parser.parse_args()

    if args.imd_instructions:
        print_imd_instructions()
        return

    current_year = datetime.now().year
    end_year = args.end_year or current_year - 1
    start_year = args.start_year or (end_year - args.years + 1)

    log.info("Downloading %s data for %d–%d", args.product.upper(), start_year, end_year)

    out_dir = RAW_DIR / args.product
    files = download_iri_temperature(start_year, end_year, out_dir)

    if args.era5:
        era5_var = {"tmax": "2m_temperature", "tmin": "2m_temperature", "rain": "total_precipitation"}
        era5_dir = RAW_DIR / "era5" / args.product
        files += download_era5(start_year, end_year, era5_dir, variable=era5_var[args.product])

    if files:
        log.info("\n✅ Downloaded %d file(s) to %s", len(files), RAW_DIR)
    else:
        log.warning("No files downloaded. Check network or run with --imd-instructions")
        print_imd_instructions()


if __name__ == "__main__":
    main()
