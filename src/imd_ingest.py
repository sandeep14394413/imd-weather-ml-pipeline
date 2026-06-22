"""
IMD Data Ingestion Module
=========================
Parses real IMD gridded binary (.GRD) and NetCDF files into a tidy
DataFrame with columns: date, state, tmax_c (or other variable).

Supported IMD products
-----------------------
  * Daily temperature max/min (.GRD binary, 1° × 1° grid, 31×31 India domain)
  * Daily rainfall (.GRD binary, 0.25° × 0.25° grid)
  * NetCDF files for any variable (ERA5 reanalysis or IMD CDO output)

Usage
-----
  from src.imd_ingest import load_imd_grd, load_netcdf, raw_dir_to_dataframe
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IMD 1° × 1° temperature grid meta (Srivastava et al. 2009)
# ---------------------------------------------------------------------------
TEMP_GRID = {
    "lat_start": 7.5,
    "lat_end": 37.5,
    "lon_start": 67.5,
    "lon_end": 97.5,
    "step": 1.0,
    "nlat": 31,
    "nlon": 31,
    "missing": 99.9,
}

# IMD 0.25° × 0.25° rainfall grid meta
RAIN_GRID = {
    "lat_start": 6.5,
    "lat_end": 38.5,
    "lon_start": 66.5,
    "lon_end": 100.0,
    "step": 0.25,
    "nlat": 129,
    "nlon": 135,
    "missing": -999.0,
}

# State centroids (lat, lon) used to extract nearest grid point
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Andhra Pradesh": (15.9, 79.7),
    "Arunachal Pradesh": (28.2, 94.7),
    "Assam": (26.2, 92.9),
    "Bihar": (25.1, 85.3),
    "Chhattisgarh": (21.3, 81.9),
    "Delhi": (28.7, 77.1),
    "Goa": (15.3, 74.0),
    "Gujarat": (22.3, 71.2),
    "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.1, 77.2),
    "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7),
    "Kerala": (10.9, 76.3),
    "Madhya Pradesh": (23.5, 77.5),
    "Maharashtra": (19.7, 75.7),
    "Manipur": (24.7, 93.9),
    "Meghalaya": (25.5, 91.4),
    "Mizoram": (23.2, 92.8),
    "Nagaland": (26.2, 94.6),
    "Odisha": (20.9, 84.5),
    "Punjab": (31.1, 75.3),
    "Rajasthan": (27.0, 74.2),
    "Sikkim": (27.5, 88.5),
    "Tamil Nadu": (11.1, 78.7),
    "Telangana": (18.1, 79.0),
    "Tripura": (23.9, 91.7),
    "Uttar Pradesh": (26.8, 80.9),
    "Uttarakhand": (30.1, 79.2),
    "West Bengal": (22.9, 87.9),
}


# ---------------------------------------------------------------------------
# Low-level binary reader
# ---------------------------------------------------------------------------

def _nearest_idx(values: np.ndarray, target: float) -> int:
    """Return index of nearest value in a 1-D sorted array."""
    return int(np.abs(values - target).argmin())


def load_imd_grd(
    filepath: Path,
    date: pd.Timestamp,
    product: Literal["tmax", "tmin", "rain"] = "tmax",
    states: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read one daily IMD .GRD binary file and return a tidy DataFrame.

    Parameters
    ----------
    filepath : Path
        Path to the .GRD file (little-endian float32, no header).
    date : pd.Timestamp
        The calendar date this file represents.
    product : str
        'tmax', 'tmin', or 'rain' — selects the correct grid metadata.
    states : list[str] | None
        Subset of STATE_CENTROIDS keys to extract. None → all states.

    Returns
    -------
    pd.DataFrame with columns: date, state, <variable>.
    """
    meta = RAIN_GRID if product == "rain" else TEMP_GRID
    nlat, nlon = meta["nlat"], meta["nlon"]
    col = "rain_mm" if product == "rain" else f"{product}_c"

    raw = np.fromfile(filepath, dtype="<f4")  # little-endian float32
    if raw.size != nlat * nlon:
        raise ValueError(
            f"{filepath.name}: expected {nlat * nlon} values, got {raw.size}. "
            "Check grid metadata or file product type."
        )
    grid = raw.reshape(nlat, nlon)  # shape: (lat, lon)
    grid[grid == meta["missing"]] = np.nan

    lats = np.arange(meta["lat_start"], meta["lat_end"] + meta["step"] / 2, meta["step"])
    lons = np.arange(meta["lon_start"], meta["lon_end"] + meta["step"] / 2, meta["step"])

    target_states = states if states else list(STATE_CENTROIDS.keys())
    rows = []
    for state in target_states:
        if state not in STATE_CENTROIDS:
            log.warning("Unknown state '%s' — skipping", state)
            continue
        lat_c, lon_c = STATE_CENTROIDS[state]
        li = _nearest_idx(lats, lat_c)
        lj = _nearest_idx(lons, lon_c)
        value = float(grid[li, lj])
        rows.append({"date": date, "state": state, col: value})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NetCDF reader (ERA5 / IMD CDO output)
# ---------------------------------------------------------------------------

def load_netcdf(
    filepath: Path,
    variable: str,
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    states: list[str] | None = None,
    kelvin_to_celsius: bool = True,
) -> pd.DataFrame:
    """
    Read a NetCDF file and extract state-level daily values.

    Parameters
    ----------
    filepath : Path
        Path to .nc or .nc4 file.
    variable : str
        NetCDF variable name, e.g. 't2m', 'TMAX', 'precip'.
    time_dim, lat_dim, lon_dim : str
        Coordinate dimension names (default: ERA5 conventions).
    states : list[str] | None
        States to extract. None → all.
    kelvin_to_celsius : bool
        If True, subtract 273.15 when the variable is in Kelvin
        (detected automatically when mean > 200).

    Returns
    -------
    pd.DataFrame with columns: date, state, <variable>.
    """
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required for .nc files: pip install netCDF4"
        ) from exc

    with nc.Dataset(filepath, "r") as ds:
        times = nc.num2date(ds[time_dim][:], ds[time_dim].units, ds[time_dim].calendar)
        dates = [pd.Timestamp(str(t)[:10]) for t in times]
        lats = ds[lat_dim][:]
        lons = ds[lon_dim][:]
        data = ds[variable][:].data.astype("float32")
        fill = getattr(ds[variable], "_FillValue", None)
        if fill is not None:
            data[data == fill] = np.nan

    if kelvin_to_celsius and np.nanmean(data) > 200:
        data -= 273.15
        log.info("Converted %s from Kelvin to Celsius", variable)

    target_states = states if states else list(STATE_CENTROIDS.keys())
    rows = []
    for t_idx, date in enumerate(dates):
        grid = data[t_idx]
        for state in target_states:
            if state not in STATE_CENTROIDS:
                continue
            lat_c, lon_c = STATE_CENTROIDS[state]
            li = _nearest_idx(np.array(lats), lat_c)
            lj = _nearest_idx(np.array(lons), lon_c)
            rows.append({"date": date, "state": state, variable: float(grid[li, lj])})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch directory loader
# ---------------------------------------------------------------------------

def raw_dir_to_dataframe(
    raw_dir: Path,
    product: Literal["tmax", "tmin", "rain"] = "tmax",
    states: list[str] | None = None,
    date_from_filename: bool = True,
) -> pd.DataFrame:
    """
    Walk *raw_dir* and parse every .GRD and .nc file found.

    File naming convention expected for .GRD files:
        TMAX_<YYYYMMDD>.GRD  or  <YYYYMMDD>.GRD

    Returns a tidy DataFrame sorted by (date, state).
    """
    frames: list[pd.DataFrame] = []
    col_map = {"tmax": "tmax_c", "tmin": "tmin_c", "rain": "rain_mm"}

    grd_files = sorted(raw_dir.glob("*.GRD")) + sorted(raw_dir.glob("*.grd"))
    nc_files = sorted(raw_dir.glob("*.nc")) + sorted(raw_dir.glob("*.nc4"))

    for grd_file in grd_files:
        try:
            date_str = "".join(filter(str.isdigit, grd_file.stem))[:8]
            date = pd.Timestamp(date_str)
        except Exception:
            log.warning("Cannot parse date from %s — skipping", grd_file.name)
            continue
        try:
            df = load_imd_grd(grd_file, date, product=product, states=states)
            frames.append(df)
            log.info("Loaded %s  → %d rows", grd_file.name, len(df))
        except Exception as exc:
            log.error("Failed to parse %s: %s", grd_file.name, exc)

    var_name = {"tmax": "t2m", "tmin": "t2m", "rain": "precip"}
    for nc_file in nc_files:
        try:
            df = load_netcdf(nc_file, variable=var_name[product], states=states)
            # Rename generic variable to our column naming convention
            df = df.rename(columns={var_name[product]: col_map[product]})
            frames.append(df)
            log.info("Loaded %s  → %d rows", nc_file.name, len(df))
        except Exception as exc:
            log.error("Failed to parse %s: %s", nc_file.name, exc)

    if not frames:
        raise FileNotFoundError(
            f"No parseable .GRD or .nc files found in {raw_dir}. "
            "Download data first with: python scripts/download_imd_data.py"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=[col_map[product]])
    combined = combined.sort_values(["date", "state"]).reset_index(drop=True)
    log.info("Total rows after merge: %d", len(combined))
    return combined
