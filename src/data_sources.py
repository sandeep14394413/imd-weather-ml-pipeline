from pathlib import Path

import requests


RAW_DIR = Path("data/raw")


def download_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return output_path


def explain_manual_imd_step() -> str:
    return (
        "IMD pages often require selecting a year/date before download. "
        "Download the selected binary or NetCDF files into data/raw, then run "
        "the parser for that product."
    )
