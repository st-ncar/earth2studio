# =============================================================================
# Imports
# =============================================================================
import dataclasses
import datetime
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from metpy.constants import dry_air_gas_constant, g
from metpy.units import units
from scipy.spatial import KDTree

from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root
from earth2studio.lexicon.mpas import MPASHybridLexicon, MPASLexicon, xtime

# =============================================================================
# Constants
# =============================================================================
# Standard lapse rate in K/m for temperature extrapolation below ground.
STANDARD_LAPSE_RATE = 0.0065 * units.K / units.m


# =============================================================================
# Base Class for MPAS Data Sources
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class _MPASBase(DataSource):
    """
    A base class for MPAS data sources providing shared functionality for
    handling unstructured grids, caching, and file I/O.

    Attributes
    ----------
    data_path : str
        A string representing the path to your MPAS data files. This path
        should be a template that can be formatted using Python's `strftime`
        convention. For example: "/path/to/data/%Y%m%d/history_%H.nc"
    grid_path : Path
        The path to the static MPAS grid definition file.
    d_lon : float, optional
        Target longitude spacing for regridding. Defaults to 0.25.
    d_lat : float, optional
        Target latitude spacing for regridding. Defaults to 0.25.
    cache_path : Path, optional
        The directory to store cached regridding indices.
    """

    data_path: str
    grid_path: Path
    d_lon: float = 0.25
    d_lat: float = 0.25
    cache_path: Path = Path(datasource_cache_root()) / "mpas_base"

    def __post_init__(self) -> None:
        """
        Post-initialization to prepare the target grid and compute or load the
        regridding indices from the cache.
        """
        self.cache_path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.data_path, list):
            self.data_path = tuple(self.data_path)

        # Use np.linspace for robust grid generation that avoids floating point
        # precision issues and guarantees endpoint inclusion.
        n_lon = int(360 / self.d_lon)
        n_lat = int(180 / self.d_lat) + 1
        self.target_lon = np.linspace(0, 360, n_lon, endpoint=False)
        self.target_lat = np.linspace(90, -90, n_lat)

        self.distance, self.indices = self._prepare_regridding_indices()

        # Create a target index for xarray's advanced indexing
        target_lon_grid, target_lat_grid = np.meshgrid(self.target_lon, self.target_lat)
        self.target_grid_index = xr.DataArray(
            self.indices,
            dims=["lat_lon"],
            coords={
                "lat": ("lat_lon", target_lat_grid.ravel()),
                "lon": ("lat_lon", target_lon_grid.ravel()),
            },
        ).set_index(lat_lon=["lat", "lon"])

        with xr.open_dataset(self.grid_path) as grid_ds:
            self.grid_ncells = grid_ds.sizes["nCells"]

    def _prepare_regridding_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates or loads cached nearest neighbor indices for regridding."""
        cache_file_name = f"{self.grid_path.stem}_{self.d_lon}x{self.d_lat}.npz"
        cached_file = self.cache_path / cache_file_name

        if cached_file.exists():
            logger.info(f"Loading cached regridding indices from {cached_file}")
            data = np.load(cached_file)
            return data["dists"], data["inds"]

        logger.info("Building KDTree from MPAS grid to compute regridding indices...")
        with xr.open_dataset(self.grid_path) as grid:
            lon_cell = grid["lonCell"]
            lat_cell = grid["latCell"]

            def process_coords(coord_da: xr.DataArray) -> np.ndarray:
                units = coord_da.attrs.get("units", "unknown").lower()
                values = coord_da.values
                if units in ["rad", "radians"]:
                    return values
                elif units in ["deg", "degrees"]:
                    return np.deg2rad(values)
                else:
                    if np.any(np.abs(values) > 2 * np.pi):
                        return np.deg2rad(values)
                    else:
                        return values

            mpas_lon_rad = process_coords(lon_cell)
            mpas_lat_rad = process_coords(lat_cell)
            mpas_xyz = self._lon_lat_to_cartesian(mpas_lon_rad, mpas_lat_rad)

        target_lon_grid, target_lat_grid = np.meshgrid(self.target_lon, self.target_lat)
        target_lon_rad = np.deg2rad(target_lon_grid.ravel())
        target_lat_rad = np.deg2rad(target_lat_grid.ravel())
        target_xyz = self._lon_lat_to_cartesian(target_lon_rad, target_lat_rad)

        kdtree = KDTree(mpas_xyz)
        logger.info("Querying tree to find nearest neighbors...")
        distance, indices = kdtree.query(target_xyz)

        logger.info(f"Saving new regridding indices to {cached_file}")
        np.savez_compressed(cached_file, dists=distance, inds=indices)
        return distance, indices

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        # Select the cells at the target grid points
        # This creates a 1D array with a multi-index (lat, lon)
        regridded_da = ds_mpas.isel(nCells=self.target_grid_index)
        # Unstack the 1D array into a 2D (or 3D) grid
        return regridded_da.unstack("lat_lon")

    @staticmethod
    def _lon_lat_to_cartesian(lon_rad: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
        """Converts lon/lat (radians) to 3D Cartesian coords for KDTree."""
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z]).T


# =============================================================================
# Pressure-Level Data Source
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class MPASPres(_MPASBase):
    """
    Custom data source for MPAS model output on pressure levels.
    """

    cache_path: Path = Path(datasource_cache_root()) / "mpas_plev"

    def __post_init__(self) -> None:
        self.lexicon = MPASLexicon
        super().__post_init__()

    @lru_cache(maxsize=16)
    def _load_and_process(
        self,
        time: datetime.datetime | np.datetime64,
        variables: tuple[str],
    ) -> xr.Dataset:
        """
        Loads, derives variables, and regrids a single time slice of
        pressure-level data in self.data_path.
        """
        source_variables = self.lexicon.required_variables(list(variables))
        logger.info(f"Requesting source variables for time {time}: {source_variables}")

        # Convert numpy.datetime64 to pandas Timestamp, which has strftime
        time_pd = pd.to_datetime(time)
        path_str = time_pd.strftime(self.data_path)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"MPAS file not found for time {time} at: {path}")

        with xtime(xr.open_dataset(path)) as ds_mpas:
            logger.info(f"Open {path}")
            ds_slice = ds_mpas.sel(time=time)

            if "time" in ds_slice.coords:
                ds_slice = ds_slice.drop_vars("time")
            # Squeeze time/initial_time lexicon.derive_variables complains about 3d boolean indices
            ds_slice = ds_slice.squeeze()

            if ds_slice.sizes.get("nCells") != self.grid_ncells:
                raise ValueError(
                    f"Grid mismatch: Grid file has {self.grid_ncells} cells, "
                    f"data file has {ds_slice.sizes['nCells']} cells."
                )

            raw_vars_to_load = [v for v in source_variables if v in ds_slice.data_vars]
            ds_filtered = ds_slice[raw_vars_to_load]
            ds_derived = self.lexicon.derive_variables(ds_filtered)
            final_vars_to_keep = [
                v for v in source_variables if v in ds_derived.data_vars
            ]
            ds_processed = ds_derived[final_vars_to_keep].load()

        logger.info("Regridding data...")
        ds_regridded = self._regrid_dataset(ds_processed)
        logger.info("Regridding complete.")
        return ds_regridded

    def _finalize_dataset(
        self, ds_regridded: xr.Dataset, variables: list[str]
    ) -> xr.DataArray:
        """Builds the final DataArray from a processed, regridded Dataset."""
        rename_dict = {self.lexicon.get_item(var): var for var in variables}
        ds_final = ds_regridded[list(rename_dict.keys())].rename(rename_dict)
        return ds_final.to_dataarray(dim="variable")

    def __call__(
        self,
        time: datetime.datetime | list[datetime.datetime] | np.ndarray,
        variables: list[str],
    ) -> xr.DataArray:
        """
        Main entry point for fetching data. Handles both single datetime requests
        (for framework runners) and lists of datetimes (for direct use).
        """
        sorted_variables = tuple(sorted(variables))

        if isinstance(time, (datetime.datetime, np.datetime64)):
            # Runner-compatible path: process a single time, return a time-unaware slice.
            ds_regridded = self._load_and_process(time, sorted_variables)
            return self._finalize_dataset(ds_regridded, variables)
        else:
            # Direct-use path: process a list of times, return a time-aware DataArray.
            results = []
            for t in time:
                ds_regridded = self._load_and_process(t, sorted_variables)
                da_slice = self._finalize_dataset(ds_regridded, variables)
                # Add time coordinate back for this slice
                da_slice = da_slice.assign_coords(time=t).expand_dims("time")
                results.append(da_slice)

            if not results:
                return xr.DataArray()
            return xr.concat(results, dim="time")


# =============================================================================
# Hybrid-Level Data Source
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class MPASHybrid(_MPASBase):
    """
    Custom data source for MPAS model output on native hybrid levels. Can also
    interpolate to pressure levels.
    """

    pressure_levels: list[int] | tuple[int, ...] | None = None
    cache_path: Path = Path(datasource_cache_root()) / "mpas_hybrid"

    def __post_init__(self) -> None:
        if isinstance(self.pressure_levels, list):
            self.pressure_levels = tuple(sorted(self.pressure_levels))
        super().__post_init__()
        self.lexicon = MPASHybridLexicon

    @lru_cache(maxsize=16)
    def _load_and_process(
        self,
        time: datetime.datetime | np.datetime64,
        variables: tuple[str],
    ) -> xr.Dataset:
        """
        Loads, processes (including vertical interpolation), and regrids data
        in self.data_path.
        """
        source_variables = self.lexicon.required_variables(list(variables))
        logger.info(f"Requesting source variables for time {time}: {source_variables}")

        # Convert numpy.datetime64 to pandas Timestamp, which has strftime
        time_pd = pd.to_datetime(time)
        path_str = time_pd.strftime(self.data_path)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"MPAS file not found for time {time} at: {path}")

        with xtime(xr.open_dataset(path)) as ds_mpas:
            logger.info(f"Open {path}")
            ds_slice = ds_mpas.sel(time=time)

            if "time" in ds_slice.coords:
                ds_slice = ds_slice.drop_vars("time")
            # Squeeze time/initial_time lexicon.derive_variables complains about 3d boolean indices
            ds_slice = ds_slice.squeeze()

            if ds_slice.sizes.get("nCells") != self.grid_ncells:
                raise ValueError(
                    f"Grid mismatch: Grid file has {self.grid_ncells} cells, "
                    f"data file has {ds_slice.sizes['nCells']} cells."
                )
            data_vars_to_load = [v for v in source_variables if v in ds_slice.data_vars]
            ds_loaded = ds_slice[data_vars_to_load].load()

        with xr.open_dataset(self.grid_path) as grid_ds:
            logger.info(f"Open {self.grid_path}")
            grid_vars_to_load = [v for v in source_variables if v in grid_ds.data_vars]
            for var_name in grid_vars_to_load:
                ds_loaded[var_name] = grid_ds[var_name].load()

        ds_derived = self.lexicon.derive_variables(ds_loaded)

        ds_processed = ds_derived
        if self.pressure_levels is not None:
            is_3d_request = any(self.lexicon.is_3d_variable(v) for v in variables)
            if is_3d_request:
                logger.info(
                    f"3D variable requested. Performing vertical interpolation to {list(self.pressure_levels)} hPa."
                )
                pressure_levels_pa = [p * 100 for p in self.pressure_levels]
                ds_processed = self._interpolate_to_pressure_levels(
                    ds_derived, pressure_levels_pa, variables
                )

        logger.info("Regridding data...")
        ds_regridded = self._regrid_dataset(ds_processed)
        logger.info("Regridding complete.")
        return ds_regridded

    def _interpolate_to_pressure_levels(
        self,
        ds: xr.Dataset,
        target_levels_pa: list[int],
        requested_variables: tuple[str],
    ) -> xr.Dataset:
        """
        Interpolates data from native hybrid levels to pressure levels, handling
        below-terrain points by either persisting surface values or extrapolating.
        """

        def vectorized_vinterp(
            data: np.ndarray,
            pressure: np.ndarray,
            targets: np.ndarray,
            interp_type: str = "linear",
        ) -> np.ndarray:
            """
            Core numpy-based interpolation function for apply_ufunc.
            data: (vert_dim,)
            pressure: (vert_dim,)
            targets: (level,)
            interp_type: 'linear' or 'log'. Default is 'linear'.
            Returns: (level,)
            """
            # Ensure pressure is increasing
            if pressure[0] > pressure[-1]:
                pressure = pressure[::-1]
                data = data[::-1]

            interp_x = pressure
            interp_target_x = targets

            if interp_type == "log":
                # Transform pressure and targets to log-space
                interp_x = np.log(pressure)
                interp_target_x = np.log(targets)
            elif interp_type != "linear":
                raise ValueError(f"Unexpected interp_type {interp_type}")

            return np.interp(interp_target_x, interp_x, data, left=np.nan, right=np.nan)

        def vectorized_top_fill(
            data: np.ndarray,
            pressure: np.ndarray,
            targets: np.ndarray,
            strategy: str,
        ) -> np.ndarray:
            """
            Based on Yessad K.'s FULL-POS IN THE CYCLE 46T1R1 OF ARPEGE/IFS
            https://www.umr-cnrm.fr/gmapdoc/IMG/pdf/ykfpos46t1r1.pdf
            Fill values above model top using variable-specific strategies.
            Returns NaN for levels that are not above model top.
            """
            data_values = np.asarray(data, dtype=np.float64)
            pressure_values = np.asarray(pressure, dtype=np.float64)
            target_values = np.asarray(targets, dtype=np.float64)

            out = np.full(target_values.shape, np.nan, dtype=np.float64)
            if data_values.size < 2 or pressure_values.size < 2:
                return out

            # Ensure ascending pressure (top of model at index 0)
            if pressure_values[0] > pressure_values[-1]:
                pressure_values = pressure_values[::-1]
                data_values = data_values[::-1]

            p1, p2 = pressure_values[0], pressure_values[1]
            v1, v2 = data_values[0], data_values[1]
            if not np.isfinite([p1, p2, v1, v2]).all() or p1 <= 0 or p2 <= 0:
                return out

            above_top = (target_values > 0) & (target_values < p1)
            if not np.any(above_top):
                return out

            x_target = np.log(
                np.clip(target_values[above_top], np.finfo(float).tiny, None)
            )
            x1 = np.log(p1)
            x2 = np.log(p2)
            # Use a small positive pressure to represent p=0 in log-pressure space.
            p_top = max(np.finfo(float).tiny, min(1.0, 0.5 * p1))
            x0 = np.log(p_top)

            if strategy == "wind":
                # Quadratic (top value linear from p1/p2)
                v_top = v1 + (v1 - v2) * (0.0 - p1) / (p1 - p2)
                coeffs = np.polyfit([x0, x1, x2], [v_top, v1, v2], deg=2)
                out[above_top] = np.polyval(coeffs, x_target)
            elif strategy == "omega":
                # Enforce omega -> 0 at p=0 with a linear profile in pressure.
                out[above_top] = v1 * (target_values[above_top] / p1)
            elif strategy == "temperature":
                # Enforce T_top = T_layer1; quadratic fit through (top, layer1, layer2).
                coeffs = np.polyfit([x0, x1, x2], [v1, v1, v2], deg=2)
                out[above_top] = np.polyval(coeffs, x_target)
            elif strategy == "geopotential":
                # Interpolate departure from a simple standard-atmosphere geopotential profile.
                rd = float(getattr(dry_air_gas_constant, "m", dry_air_gas_constant))
                t_std = 255.0
                p0_std = 101325.0

                def standard_geopotential(p: np.ndarray | float) -> np.ndarray:
                    p_arr = np.asarray(p, dtype=np.float64)
                    return (
                        -rd
                        * t_std
                        * np.log(np.clip(p_arr, np.finfo(float).tiny, None) / p0_std)
                    )

                phi_std_1 = standard_geopotential(p1)
                phi_std_2 = standard_geopotential(p2)
                e1 = v1 - phi_std_1
                e2 = v2 - phi_std_2
                if np.isclose(p1, p2):
                    e_top = e1
                else:
                    e_top = e1 + (e1 - e2) * (0.0 - p1) / (p1 - p2)

                coeffs = np.polyfit([x0, x1, x2], [e_top, e1, e2], deg=2)
                departure = np.polyval(coeffs, x_target)
                out[above_top] = departure + standard_geopotential(
                    target_values[above_top]
                )
            elif strategy == "humidity":
                out[above_top] = v1
            else:
                raise ValueError(f"Unexpected top-fill strategy {strategy}")

            return out

        vars_to_interp = {
            self.lexicon.get_derived_name(v)
            for v in requested_variables
            if self.lexicon.is_3d_variable(v)
        }
        moisture_vars_to_interp = {
            self.lexicon.get_derived_name(v)
            for v in requested_variables
            if self.lexicon.is_3d_variable(v) and re.sub(r"\d+$", "", v) in {"q", "r"}
        }

        # Create a 1D DataArray for target pressure levels
        target_levels_pa_da = xr.DataArray(
            target_levels_pa * units.Pa,
            dims=["level"],
            coords={"level": [p / 100 for p in target_levels_pa]},
        )
        target_levels_pa_da["level"].attrs["units"] = "hPa"

        interpolated_vars = {}
        for name, da in ds.data_vars.items():
            is_main = "nVertLevels" in da.dims
            is_staggered = "nVertLevelsP1" in da.dims

            if (is_main or is_staggered) and name in vars_to_interp:
                logger.info(f"Interpolating variable: {name}")
                pressure_field = ds["pressure"] if is_main else ds["pressure_on_w"]
                vert_dim = "nVertLevels" if is_main else "nVertLevelsP1"
                # is_ascending means level 0 is near sfc; last level (-1) is at top
                is_ascending = pressure_field.isel(
                    {"nCells": 0, vert_dim: 0}
                ) > pressure_field.isel({"nCells": 0, vert_dim: -1})
                lowest_model_level = 0 if is_ascending else -1

                # Select log or linear interpolation in pressure.
                # log for geopotential and wind
                # linear for temperature, moisture, etc.
                if name in [
                    "geopotential",
                    "uReconstructMeridional",
                    "uReconstructZonal",
                ]:
                    logger.info(f"Using ln(p) interpolation for {name}")
                    interp_kwargs = {"interp_type": "log"}
                else:
                    logger.info(f"Using linear p interpolation for {name}")
                    interp_kwargs = {"interp_type": "linear"}

                # 1. Perform vectorized interpolation
                interp_da_nocoords = xr.apply_ufunc(
                    vectorized_vinterp,
                    da,
                    pressure_field,
                    target_levels_pa_da,  # Pass the 1D target levels
                    kwargs=interp_kwargs,
                    input_core_dims=[[vert_dim], [vert_dim], ["level"]],
                    output_core_dims=[["level"]],  # Output has 'level' dim
                    exclude_dims={vert_dim, "level"},
                    vectorize=True,
                    output_dtypes=[da.dtype],
                )
                interp_da = interp_da_nocoords.assign_coords(
                    level=target_levels_pa_da.level
                )

                # 2. Get mask of all pts that need filling.
                nan_mask = interp_da.isnull()
                top_pressure = pressure_field.min(dim=vert_dim)
                bottom_pressure = pressure_field.max(dim=vert_dim)
                above_top_mask = nan_mask & (target_levels_pa_da < top_pressure)
                below_surface_mask = nan_mask & (target_levels_pa_da > bottom_pressure)

                is_wind = name in {"uReconstructMeridional", "uReconstructZonal"}
                is_humidity = name in moisture_vars_to_interp

                # Apply variable-specific top-of-model fill before below-ground handling.
                top_strategy = None
                if is_wind:
                    top_strategy = "wind"
                elif name == "pressure_vertical_velocity":
                    top_strategy = "omega"
                elif name == "geopotential":
                    top_strategy = "geopotential"
                elif name == "temperature":
                    top_strategy = "temperature"
                elif is_humidity:
                    top_strategy = "humidity"

                if top_strategy is not None:
                    top_fill_nocoords = xr.apply_ufunc(
                        vectorized_top_fill,
                        da,
                        pressure_field,
                        target_levels_pa_da,
                        kwargs={"strategy": top_strategy},
                        input_core_dims=[[vert_dim], [vert_dim], ["level"]],
                        output_core_dims=[["level"]],
                        exclude_dims={vert_dim, "level"},
                        vectorize=True,
                        output_dtypes=[da.dtype],
                    )
                    top_fill_da = top_fill_nocoords.assign_coords(
                        level=target_levels_pa_da.level
                    )
                    interp_da = xr.where(above_top_mask, top_fill_da, interp_da)

                surface_pressure = ds["surface_pressure"]

                # Considered FULL-POS CYCLE 46T1R1
                # But there seems to be an error in geopotential extrapolation below surface.
                # It seems to work if you flip the pressures inside the logarithm.
                # Or maybe pi_L is not what I think. Is it any layer in target layers or
                # just the bottom layer? Plus, getting surface temperature in high topography is hard (Eqn (2)-(5)).

                # Follow Trenberth Eqn (15) The order of pressures inside the logarithm make sense.
                ln_pressure_ratio = np.log(target_levels_pa_da / surface_pressure)
                alpha = STANDARD_LAPSE_RATE * dry_air_gas_constant / g
                y = alpha * ln_pressure_ratio
                if name == "temperature":
                    surface_temperature = ds["surface_temperature"]
                    # Trenberth et al. Eqn (16-19)
                    surface_height = ds["geopotential_at_surface"] / g
                    t_0 = surface_temperature + surface_height * STANDARD_LAPSE_RATE
                    orog = surface_height >= 2000 * units.m
                    medium_orog = (surface_height >= 2000 * units.m) & (
                        surface_height <= 2500 * units.m
                    )
                    high_orog = surface_height > 2500 * units.m

                    # Calculate Tpl (Plateau Temperature)
                    Tpl = xr.where(
                        t_0 >= 298 * units.K, xr.DataArray(298) * units.K, t_0
                    )  # Eqn (18)

                    # Update t_0 based on orography
                    # medium orography formula (Eq 19b)
                    t_0_medium_orog = (
                        0.002
                        / units.m
                        * (
                            (2500 * units.m - surface_height) * t_0
                            + (surface_height - 2000 * units.m) * Tpl
                        )
                    )

                    t_0 = xr.where(high_orog, Tpl, t_0)  # Eqn (18)
                    t_0 = xr.where(medium_orog, t_0_medium_orog, t_0)  # Eqn (19b)

                    # Calculate alpha_orog (Eq 17)
                    alpha_orog = (
                        dry_air_gas_constant
                        * (t_0 - surface_temperature)
                        / ds["geopotential_at_surface"]
                    )

                    # Apply alpha_orog where 'orog' is True, keep original alpha elsewhere
                    alpha = xr.where(orog, alpha_orog, alpha)

                    # Line after Eqn (19b) if t_0 < surface_temperature set alpha to zero.
                    alpha = xr.where(t_0 < surface_temperature, xr.DataArray(0), alpha)

                    # Final Extrapolation (Eq 16)
                    y = alpha * ln_pressure_ratio
                    extrap_values = surface_temperature * (1 + y + y**2 / 2 + y**3 / 6)

                    # Final Merge (below-ground only)
                    final_da = xr.where(
                        below_surface_mask, extrap_values.metpy.dequantify(), interp_da
                    )

                elif name == "geopotential":
                    surface_geopotential = ds["geopotential_at_surface"]
                    surface_temperature = ds["surface_temperature"]
                    low_temp = (
                        surface_temperature < 255 * units.K
                    )  # Eqn (14.3) Trenberth says "below ground geopotential is treated as for mslp".
                    t_0 = (
                        surface_temperature
                        + STANDARD_LAPSE_RATE * surface_geopotential / g
                    )  # Eqn (13) temperature at zero geopotential (msl)
                    high_temp = (surface_temperature > 290.5 * units.K) & (
                        t_0 > 290.5 * units.K
                    )

                    # Calculate reduced alpha Eqn (14.1)
                    alpha_reduced = (
                        dry_air_gas_constant
                        / surface_geopotential
                        * (290.5 * units.K - surface_temperature)
                    )
                    mask = (surface_temperature <= 290.5 * units.K) & (
                        t_0 > 290.5 * units.K
                    )
                    alpha = xr.where(mask, alpha_reduced, alpha)  # line before Eqn (13)

                    alpha = xr.where(
                        high_temp, xr.DataArray(0), alpha
                    )  # line before Eqn (14.2)
                    surface_temperature = xr.where(
                        high_temp,
                        (surface_temperature + 290.5 * units.K) / 2,
                        surface_temperature,
                    )  # line before Eqn (14.2)
                    surface_temperature = xr.where(
                        low_temp,
                        (surface_temperature + 255 * units.K) / 2,
                        surface_temperature,
                    )  # line before Eqn (14.3)
                    # Trenberth et al. Eqn. (15)
                    extrap_values = (
                        surface_geopotential
                        - dry_air_gas_constant
                        * surface_temperature
                        * ln_pressure_ratio
                        * (1 + y / 2 + y**2 / 6)
                    )
                    final_da = xr.where(
                        below_surface_mask, extrap_values.metpy.dequantify(), interp_da
                    )

                else:
                    # For all other variables, fill below-ground by persisting surface value
                    surface_value = da.isel(
                        {vert_dim: lowest_model_level}
                    ).metpy.dequantify()
                    final_da = xr.where(below_surface_mask, surface_value, interp_da)

                interpolated_vars[name] = final_da

            elif not is_main and not is_staggered:
                # This is a 2D variable, dequantify it as well
                interpolated_vars[name] = da.metpy.dequantify()

        interp_ds = xr.Dataset(interpolated_vars, attrs=ds.attrs)
        return interp_ds

    def _finalize_dataset(
        self, ds_regridded: xr.Dataset, variables: list[str]
    ) -> xr.DataArray:
        """Builds the final DataArray from a processed, regridded Dataset."""
        vars_to_build = {}
        for var in variables:
            source_name = self.lexicon.get_derived_name(var)
            if source_name not in ds_regridded:
                raise KeyError(
                    f"Requested variable '{var}' (mapped to '{source_name}') could not be found or derived."
                )

            da = ds_regridded[source_name]
            match = re.fullmatch(r"([a-zA-Z]+)([0-9]+)", var)
            if match and "level" in da.coords:
                level = int(match.group(2))
                vars_to_build[var] = da.sel(level=level, method="nearest").drop_vars(
                    "level"
                )
            else:
                vars_to_build[var] = da

        ds_final = xr.Dataset(vars_to_build)
        return ds_final.to_dataarray(dim="variable")

    def __call__(
        self,
        time: datetime.datetime | list[datetime.datetime] | np.ndarray,
        variables: list[str],
    ) -> xr.DataArray:
        """
        Main entry point for fetching data. Handles both single datetime requests
        (for framework runners) and lists of datetimes (for direct use).
        """
        sorted_variables = tuple(sorted(variables))

        if isinstance(time, (datetime.datetime, np.datetime64)):
            # Runner-compatible path: process a single time, return a time-unaware slice.
            ds_regridded = self._load_and_process(time, sorted_variables)
            return self._finalize_dataset(ds_regridded, variables)
        else:
            # Direct-use path: process a list of times, return a time-aware DataArray.
            results = []
            for t in time:
                ds_regridded = self._load_and_process(t, sorted_variables)
                da_slice = self._finalize_dataset(ds_regridded, variables)
                # Add time coordinate back for this slice
                da_slice = da_slice.assign_coords(time=t).expand_dims("time")
                results.append(da_slice)

            if not results:
                return xr.DataArray()
            return xr.concat(results, dim="time")
