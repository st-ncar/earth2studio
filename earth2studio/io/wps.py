# =============================================================================
# Imports
# =============================================================================
import re
import struct
from collections import OrderedDict
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from metpy.constants import g

from earth2studio.data import LandSeaMask, SurfaceGeoPotential
from earth2studio.io import IOBackend


# =============================================================================
# Main IO Backend Class
# =============================================================================
class WPSBackend(IOBackend):
    """
    An IOBackend that writes the final time step of a forecast to a dynamically
    named WPS intermediate binary file. It also writes landseamask and surface height
    to the final output.
    """

    VARIABLE_MAP = {
        # Maps e2s name to (WPS Field Name, Units, Description, WPS Level Code)
        "t": ("TT", "K", "Temperature", -1),
        "z": ("GHT", "m", "Geopotential Height", -1),
        "u": ("UU", "m s-1", "Zonal Wind", -1),
        "v": ("VV", "m s-1", "Meridional Wind", -1),
        "q": ("SPECHUMD", "kg kg-1", "Specific Humidity", -1),
        "r": ("RH", "%", "Relative Humidity", -1),
        "w": ("WW", "Pa s-1", "Vertical Velocity", -1),
        "msl": ("PMSL", "Pa", "Sea Level Pressure", 201300.0),
        "lsm": ("LANDSEA", "proportion", "Land/Sea Mask", 200100.0),
        "t2m": ("TT", "K", "2-meter Temperature", 200100.0),
        "u10m": ("UU", "m s-1", "10-meter U-wind", 200100.0),
        "v10m": ("VV", "m s-1", "10-meter V-wind", 200100.0),
        "tp": ("PRECIP", "kg m-2", "Total Precipitation 6-h", 200100.0),
    }

    # Format strings based on the reference Fortran source.
    # Using '>' for big-endian byte order to match the WPS standard.
    HEADER_FORMAT = "> 24s f 32s 9s 25s 46s f i i i"
    PROJECTION_FORMAT_LATLON = "> 8s f f f f f"

    def __init__(
        self,
        path: str | Path,
        model_source: str = "earth2studio",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the WPSBackend.

        Parameters
        ----------
        path : str | Path
            The output directory where the final forecast file will be saved.
        model_source : str, optional
            The name of the source model, used for generating the output filename.
            Defaults to "earth2studio".
        """
        super().__init__()
        self.output_dir = Path(path)
        self.model_source = model_source
        self.final_data_package: tuple | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_record(self, file_handle: IO[bytes], data: bytes) -> None:
        """Writes a Fortran-style record (marker, data, marker)."""
        marker = struct.pack(">i", len(data))
        file_handle.write(marker)
        file_handle.write(data)
        file_handle.write(marker)

    def write(
        self,
        data_list: list[np.ndarray] | xr.Dataset | xr.DataArray,
        coords: OrderedDict[str, np.ndarray] | None = None,
        array_name: str | list[str] | None = None,
    ) -> None:
        """
        Stores the data. Now supports xarray objects directly.
        """
        # 1. Handle xarray.Dataset
        if isinstance(data_list, xr.Dataset):
            ds = data_list
            array_name = list(ds.data_vars.keys())
            # Convert to list of numpy arrays
            processed_data = [ds[var].values for var in array_name]
            # Extract coords as an OrderedDict
            coords = OrderedDict({d: ds[d].values for d in ds.dims})
            data_list = processed_data

        # 2. Handle xarray.DataArray
        elif isinstance(data_list, xr.DataArray):
            da = data_list
            array_name = [da.name] if da.name else ["variable"]
            coords = OrderedDict({d: da[d].values for d in da.dims})
            data_list = [da.values]

        # 3. Fallback to standard earth2studio list/ndarray input
        if coords is None or array_name is None:
            raise ValueError(
                "coords and array_name are required if data_list is not an xarray object."
            )

        # Store for the .close() method logic
        self.final_data_package = (data_list, coords, array_name)

    def _write_complete_field(
        self,
        f: IO[bytes],
        da: xr.DataArray,
        hdate: str,
        field_name: str,
        units: str,
        desc: str,
        xlvl: float,
        nx: int,
        ny: int,
        coords: OrderedDict[str, np.ndarray],
        xfcst: float,
    ) -> None:
        """Writes all five Fortran records for a single 2D field."""
        # --- Record 1: Version ---
        self._write_record(f, struct.pack(">i", 5))

        # --- Record 2: Main Header ---
        header_data = struct.pack(
            self.HEADER_FORMAT,
            hdate.ljust(24).encode("utf-8"),
            xfcst,
            self.model_source.ljust(32).encode("utf-8"),
            field_name.ljust(9).encode("utf-8"),
            units.ljust(25).encode("utf-8"),
            desc.ljust(46).encode("utf-8"),
            xlvl,
            nx,
            ny,
            0,  # iproj = 0
        )
        self._write_record(f, header_data)

        # --- Record 3: Projection ---
        lat = coords["lat"]
        lon = coords["lon"]
        proj_data = struct.pack(
            self.PROJECTION_FORMAT_LATLON,
            "SWCORNER".ljust(8).encode("utf-8"),
            float(lat[0]),
            float(lon[0]),
            float(lat[1] - lat[0] if len(lat) > 1 else 0.0),
            float(lon[1] - lon[0] if len(lon) > 1 else 0.0),
            6371229.0,  # Earth radius in m
        )
        self._write_record(f, proj_data)

        # --- Record 4: Wind Flag ---
        # is_wind_grid_rel = .FALSE. -> 0 for Earth-relative winds
        self._write_record(f, struct.pack(">i", 0))

        # --- Record 5: Data Payload ---
        flat_data = da.values.astype(">f4")
        self._write_record(f, flat_data.tobytes())

    def close(self) -> None:
        """
        Writes the final stored forecast step, injecting static fields,
        to a dynamically named binary file.
        """
        if self.final_data_package is None:
            logger.warning("WPSBackend closing, but no data was provided to write.")
            return

        data_list, coords, array_name = self.final_data_package

        if isinstance(array_name, str):
            array_name = [array_name]

        valid_time = pd.to_datetime(coords["time"][0])

        # --- Create dynamic filename ---
        model_prefix = self.model_source.upper()[0:4]
        time_str = valid_time.strftime("%Y-%m-%d_%H")
        filename = f"{model_prefix}:{time_str}"
        output_path = self.output_dir / filename

        if output_path.exists():
            logger.warning(
                f"Output file {output_path} already exists and will be overwritten."
            )
            output_path.unlink()

        # --- Process and write the data ---
        processed_data = [
            d.cpu().numpy() if hasattr(d, "cpu") else d for d in data_list
        ]
        data_vars = {
            name: (list(coords.keys()), data)
            for name, data in zip(array_name, processed_data)
        }
        ds = xr.Dataset(data_vars, coords=coords)

        logger.info("Adding static fields.")
        z = SurfaceGeoPotential()([pd.Timestamp(0)])
        lsm = LandSeaMask()([pd.Timestamp(0)])
        ds["z"] = z.squeeze().expand_dims(time=ds.time)  # divide by g later
        ds["lsm"] = lsm.squeeze().expand_dims(time=ds.time)

        hdate = valid_time.strftime("%Y-%m-%d_%H:%M:%S")

        xfcst: float
        if "lead_time" in coords:
            lead_time_delta = coords["lead_time"][0]
            xfcst = lead_time_delta / np.timedelta64(1, "h")
        else:
            xfcst = 0.0

        logger.info(
            f"Writing final forecast step for time {hdate} (F{xfcst:03.0f}) to {output_path}"
        )

        with open(output_path, "ab") as f:
            for var_name_str in ds.data_vars:
                # (Logic to find base_var and write field is the same as before)
                base_var = None
                if str(var_name_str) in self.VARIABLE_MAP:
                    base_var = str(var_name_str)
                else:
                    match = re.match(r"([a-zA-Z]+)", str(var_name_str))
                    if match and match.group(1) in self.VARIABLE_MAP:
                        base_var = match.group(1)

                if not base_var:
                    logger.warning(f"No WPS mapping for '{var_name_str}', skipping.")
                    continue

                field_name, units, desc, xlvl_code = self.VARIABLE_MAP[base_var]
                da = ds[var_name_str]

                if "time" in da.dims:
                    da = da.squeeze("time")
                if "lat" in da.dims and "lon" in da.dims:
                    da = da.transpose(..., "lat", "lon")
                if base_var == "tp":
                    # convert m to kg / m**2
                    da = da * 1000.0
                if base_var == "z":
                    da = da / g.magnitude

                nx, ny = da.sizes.get("lon", 1), da.sizes.get("lat", 1)

                final_xlvl: float
                if str(var_name_str) == "z":
                    field_name = "SOILGHT"
                    final_xlvl = 200100.0
                elif xlvl_code == -1:
                    level_match = re.search(r"(\d+)$", str(var_name_str))
                    if level_match:
                        level_hpa = float(level_match.group(1))
                        final_xlvl = level_hpa * 100.0
                    else:
                        continue
                else:
                    final_xlvl = xlvl_code

                self._write_complete_field(
                    f,
                    da,
                    hdate,
                    field_name,
                    units,
                    desc,
                    final_xlvl,
                    nx,
                    ny,
                    coords,
                    xfcst,
                )

        logger.info(f"WPSBackend closed. Final output: {output_path}")
