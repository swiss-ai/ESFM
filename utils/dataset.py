# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import os
import glob
import pickle
import random
import itertools
from typing import Optional, List, Callable, Any, Dict, Sequence, Union
from datetime import datetime, timedelta
from copy import deepcopy

import torch
import cftime
import mmnpz
import numpy as np
import xarray as xr
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from esfm.normalisation import load_normalization_stats


d_srf_abr2full = {"2t": "2m_temperature", 
                  "10u": "10m_u_component_of_wind", 
                  "10v": "10m_v_component_of_wind", 
                  "msl": "mean_sea_level_pressure", 
                  "tp": "total_precipitation",
                  "tp_log": "total_precipitation_log",
                  "tp_mswep": "total_precipitation_MSWEP",
                  "tp_mswep_log": "total_precipitation_MSWEP_log",
                  "pe": "potential_evaporation",
                  "e": "evaporation",
                  "r": "runoff",
                  "swvl": "volumetric_soil_water_layer",
                  "swvl_1": "volumetric_soil_water_layer_1",
                  "swc": "soil_water_content",
                  "tws": "terrestrial_water_storage",
                  "tws_gou": "terrestrial_water_storage_Gou",
                  "tws_itsg": "terrestrial_water_storage_ITSG",
                  "slhf": "surface_latent_heat_flux",
                  "sshf": "surface_sensible_heat_flux",
                  "ssr": "surface_net_solar_radiation", 
                  "str": "surface_net_thermal_radiation",
                  "ssrd": "surface_solar_radiation_downwards",
                  "strd": "surface_thermal_radiation_downwards",
                  "tsr": "top_net_solar_radiation",
                  "ttr": "top_net_thermal_radiation",
                  "tisr": "toa_incident_solar_radiation",
                  "sst": "sea_surface_temperature",
                  "ci": "sea_ice_cover",
                  "co2": "global_CO2",
                  "pe_6hr": "potential_evaporation_6hr",
                  "e_6hr": "evaporation_6hr",
                  "r_6hr": "runoff_6hr",
                  "tp_6hr": "total_precipitation_6hr",
                  "slhf_6hr": "surface_latent_heat_flux_6hr",
                  "sshf_6hr": "surface_sensible_heat_flux_6hr",
                  "ssr_6hr": "surface_net_solar_radiation_6hr", 
                  "str_6hr": "surface_net_thermal_radiation_6hr",
                  "ssrd_6hr": "surface_solar_radiation_downwards_6hr",
                  "strd_6hr": "surface_thermal_radiation_downwards_6hr",
                  "tsr_6hr": "top_net_solar_radiation_6hr",
                  "ttr_6hr": "top_net_thermal_radiation_6hr",
                  "sf_6hr": "snowfall_6hr",
                  "ir_mod": "MOD05_L2_avg_IR", 
                  "nir_mod": "MOD05_L2_avg_NIR",
                  "ir_myd": "MYD05_L2_avg_IR",
                  "nir_myd": "MYD05_L2_avg_NIR",
                  "u_10m_cosmo": "U_10M",
                  "v_10m_cosmo": "V_10M",
                  "vmax_10m": "VMAX_10M",
                  "2d": "TD_2M",
                  "sp": "PS",
                  "clct_cosmo": "CLCT",
                  "alhfl_s": "ALHFL_S",
                  "ashfl_s": "ASHFL_S",
                  "asob_s": "ASOB_S",
                  "athb_s": "ATHB_S",
                  "tp_cosmo": "TOT_PREC",
                  "tcwv": "TQV",
                  "w_snow_cosmo": "W_SNOW",
                  "2q": "2m_specific_humidity",
                  "pa": "air_pressure",
                  "dt": "dew_point_temperature",
                  "wd": "wind_from_direction",
                  "ws": "wind_speed",
                  "ts": "time_shift_minutes"
                  }
d_static_abr2full = {"lsm": "land_sea_mask", 
                     "z": "geopotential_at_surface", 
                     "slt": "soil_type",
                     "z_cosmo": "FI",
                     "hsurf": "HSURF",
                     "p0fl": "P0FL"} ## nonexisting variable in wb2: "slt": "soil_type"
d_atmos_abr2full = {"z": "geopotential", 
                    "u": "u_component_of_wind", 
                    "v": "v_component_of_wind", 
                    "t": "temperature", 
                    "q": "specific_humidity", 
                    "w": "vertical_velocity",}

d_srf_full2abr = {v: k for k, v in d_srf_abr2full.items()}
d_static_full2abr = {v: k for k, v in d_static_abr2full.items()}
d_atmos_full2abr = {v: k for k, v in d_atmos_abr2full.items()}

def _merge_time_ordered(ds_list: list, use_dask_chunks: bool = True):
    """Merge a list of time-ordered DataArrays/Datasets along time axis.
    
    Args:
        ds_list: List of xarray objects to merge along time
        use_dask_chunks: If True, convert chunks=None to dask chunks before concat
                         to avoid OOM on large concatenations. Default True.
    """
    if not ds_list:
        raise ValueError("ds_list is empty")
    if len(ds_list) == 1:
        return ds_list[0]
    
    # Convert to dask chunks if requested to avoid OOM during concat
    if use_dask_chunks:
        ds_list = [
            d.chunk({'time': 'auto'}) if hasattr(d, 'chunk') and d.chunks is None 
            else d 
            for d in ds_list
        ]
    
    return xr.concat(ds_list, dim="time")


def _subset_time_window(data_obj, inds, lead_time_h):
    """Subset time-dependent data to requested inds plus forecast history/target margin."""
    if inds is None or "time" not in data_obj.dims:
        return data_obj

    try:
        inds_arr = np.asarray(inds, dtype="datetime64[ns]")
    except Exception:
        return data_obj

    if inds_arr.size == 0:
        return data_obj

    t_min = inds_arr.min() - np.timedelta64(int(lead_time_h), "h")
    t_max = inds_arr.max() + np.timedelta64(int(lead_time_h), "h")
    return data_obj.sel(time=slice(t_min, t_max))


def ensure_contiguous(data):
    """Recursively ensure numpy arrays have positive strides."""
    if isinstance(data, np.ndarray):
        # Check for negative strides which PyTorch doesn't support
        if any(s < 0 for s in data.strides):
            return data.copy()
        return data
    elif isinstance(data, dict):
        return {k: ensure_contiguous(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_contiguous(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(ensure_contiguous(v) for v in data)
    return data

def _init_prng():
    """
    Initializes a separate pseudo-random number generator (PRNG) for each worker.
    
    This method ensures that each worker in a multi-worker data loading setup has its own
    PRNG instance, initialized with a unique seed. The seed is derived from `torch.initial_seed()`,
    which is specific to each worker. This approach guarantees reproducibility and prevents
    workers from sharing the same random number sequence.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        # Single-process data loading
        seed = torch.initial_seed() % 2**32
    else:
        # In worker process
        seed = torch.initial_seed() % 2**32
    prng = np.random.RandomState(seed)
    return prng

class ScalarCO2Mapper:
    def __init__(
            self,
            co2_path: str = '/capstor/store/cscs/swissai/a122/hydrological_data/global_annual_CO2.csv',
            co2_fullname: str='global_CO2',
            lead_time_h: int=6,
            inds = None,
            ):
        self.co2_fullname = co2_fullname

        co2_df = pd.read_csv(co2_path)
        co2_df.columns = ["year", "CO2"]
        time_base = np.array([np.datetime64(f'{year}-07-02T00:00:00.000000000') for year in co2_df["year"]]) # 2nd of July is the middle of the year
        # add timesteps at the ends of range (the need for this depends on the inds definition)
        ## check if inds increment regularly as lead_time_h:
        inds_dt = np.asarray(inds[1:]) - np.asarray(inds[:-1])
        if np.all(inds_dt == np.timedelta64(lead_time_h, 'h')):
            inds = np.concatenate([[inds[0] - np.timedelta64(lead_time_h, 'h')], inds, [inds[-1] + np.timedelta64(lead_time_h, 'h')]])
        else:
            ## inds are irregular, we need to first create a -leadtime +leadtime for each ind, and then drop duplicates:
            inds_ = np.concatenate([[i - np.timedelta64(lead_time_h, 'h'), i, i + np.timedelta64(lead_time_h, 'h')] for i in inds])
            # now drop duplicates and sort
            inds_ = np.unique(inds_)
            inds = natsorted(inds_)

        # Interpolate from yearly to hourly
        co2_df.index = time_base
        co2_df.drop('year', axis=1, inplace=True)
        co2_df = pd.concat([co2_df, pd.DataFrame(index=inds, columns=['CO2'], dtype=float)]).sort_index()
        co2_df.loc[:, 'time'] = co2_df.index
        co2_df = co2_df.drop_duplicates(subset='time').drop('time', axis=1)
        co2_df.interpolate(inplace=True)
        self.co2_df = co2_df
        

    def getitem(self, t: list = None, lat: list = None, lon: list = None, inference=False):
        if not isinstance(t, list) and not inference:
            t = [t]
        co2_data = self.co2_df.loc[t, "CO2"].to_numpy()[:, np.newaxis, np.newaxis].astype(np.float32)
        co2_data = np.broadcast_to(co2_data, (len(t), len(lat), len(lon))).copy()
        co2 = xr.DataArray(
                co2_data,
                coords={
                    "time": t,
                    "latitude": lat,
                    "longitude": lon
                },
                dims=["time", "latitude", "longitude"],
                name=self.co2_fullname
            )
        return co2


class WeatherBench2Raw(Dataset):
    def __init__(
        self, 
        name='era5',
        path: str = '/capstor/store/cscs/ERA5/weatherbench2_original', 
        extended_path: dict = None, # key=variable short name, value=path to dataset that contains this extra variable
        extended_vars: list = None, # list of the variables (short name) from extended_dataset to include in original dataset
        stats_path: str = 'esfm/normalization_stats_1979_2021.json',
        inds = None, 
        str_task: str = 'forecast', 
        dict_vars: dict = None, 
        surf_vars: list[str] = None,
        static_vars: list[str] = None,
        atmos_vars: list[str] = None,
        atmos_levels = np.asarray([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int32),
        dict_stats: Optional[dict[str, tuple[float, float]]] = None, 
        co2_path: str = '/capstor/store/cscs/swissai/a122/hydrological_data/global_annual_CO2.csv',
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        lead_time_h: int = 6, # lead time in hours for the forecast task
        with_cache: bool = True, # get indices from cache files
        **kwargs,
    ):
        '''Defining surf_vars, static_vars, atmos_vars will overwrite dict_vars. 
        variable_name_mapping is ignored since all other datasets must conform to ERA5 convention.'''
        self.name = name
        self.path = path
        self.inds = inds
        self.d_ind_pairs = {}
        self.str_task = str_task
        self.dict_vars = dict_vars
        self.atmos_levels = atmos_levels
        if isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)
        self.dict_stats = dict_stats
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution
        self.ds = xr.open_zarr(path, chunks=None)
        self.dict_ds_extended = dict()
        self.lead_time_h = lead_time_h
        self.with_cache = with_cache
        
        if len(self.ds.latitude) == 721:
            self.lat = self.ds.latitude.values[:-1] ## get only 720 out of the 721 latitudes
        else:
            self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values
        self.lead_time_x_hist = kwargs.pop('lead_time_x_hist', lead_time_h) # lead time in hours for the reconstruction task

        self.locations, self.scales = load_normalization_stats(stats_path)
        
        if extended_path:
            for var in extended_vars:
                self.dict_ds_extended[d_srf_abr2full[var]] = xr.open_zarr(extended_path[var], chunks=None)[d_srf_abr2full[var].replace('_log', '')].sel(latitude=self.lat, longitude=self.lon) # chunks=None as for self.ds: no dask in the workers
            # self.ds = self.ds.assign({var: self.dict_ds_extended[var] for var in extended_vars}) # remove because not lazy load

        
        if self.inds is None: 
            raise AssertionError("dataset indices not provided.")
        self.len_dataobj = len(self.inds) ## will be later overwritten.
        if self.dict_vars is None:
            self.dict_vars = {
                'surf_vars': ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"),
                'static_vars': ("land_sea_mask", "geopotential_at_surface", "soil_type"),
                'atmos_vars': ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity")
            }
        self.ds = self.ds.sel(level=self.atmos_levels)
        self.ds = self.ds.sel(latitude=self.lat)
        self.ds = self.ds.sel(longitude=self.lon)
        if surf_vars is not None:
            self.surf_vars = dict()
            for k in surf_vars:
                if d_srf_abr2full[k] in self.ds.data_vars:
                    self.surf_vars[k] = self.ds[d_srf_abr2full[k]]
                elif '_log' in d_srf_abr2full[k]:
                    full_name_ = d_srf_abr2full[k].replace('_log', '')
                    if full_name_ in self.ds.data_vars:
                        self.surf_vars[k] = self.ds[full_name_]
                        print(f'Using {full_name_} instead of {d_srf_abr2full[k]} from ds for dataset.')
                else:
                    if k != 'co2':
                        self.surf_vars[k] = self.dict_ds_extended[d_srf_abr2full[k]]

            self.dict_vars['surf_vars'] = tuple([d_srf_abr2full[k] for k in surf_vars])
            assert len(self.dict_vars['surf_vars']) == len(set(self.dict_vars['surf_vars'])), "You should not have duplicated surface variables (based on full names)"
        else:
            self.surf_vars = {d_srf_full2abr[var]: self.ds[var] for var in self.dict_vars['surf_vars']} ## respecting the abbreviations from Aurora implementation for dict keys
        if static_vars is not None:
            self.static_vars = {k: self.ds[d_static_abr2full[k]] for k in static_vars}
            self.dict_vars['static_vars'] = tuple([d_static_abr2full[k] for k in static_vars])
        else:
            self.static_vars = {d_static_full2abr[var]: self.ds[var] for var in self.dict_vars['static_vars']}
        if atmos_vars is not None:
            self.atmos_vars = {k: self.ds[d_atmos_abr2full[k]] for k in atmos_vars}
            self.dict_vars['atmos_vars'] = tuple([d_atmos_abr2full[k] for k in atmos_vars])
        else:
            self.atmos_vars = {d_atmos_full2abr[var]: self.ds[var] for var in self.dict_vars['atmos_vars']}
        if self.str_task == '6h-forecast':
            self._prepare_inds_for_forecast(lead_time_h=6) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)
        elif self.str_task == 'forecast' and lead_time_h != 0:
            self._prepare_inds_for_forecast(lead_time_h=lead_time_h) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)

        if 'co2' in surf_vars: # must be executed after defining self.lat and self.lon
            self.co2_mapper = ScalarCO2Mapper(co2_path = co2_path,
                                  co2_fullname = d_srf_abr2full['co2'],
                                  lead_time_h = self.lead_time_h,
                                  inds = self.inds
                                  )
        
        # timestamp → position map (built once per worker) 
        times_ns = self.ds.time.values.astype("datetime64[ns]")
        self._time2idx = {int(t): idx for idx, t in enumerate(times_ns)}
        self._time2idx_extended = dict()
        for var in self.dict_vars['surf_vars']:
            if var not in self.ds.data_vars and var in self.dict_ds_extended:
                ds_ext = self.dict_ds_extended[var]
                times_ns_ext = ds_ext.time.values.astype("datetime64[ns]")
                self._time2idx_extended[var] = {int(t): idx for idx, t in enumerate(times_ns_ext)}

        # static variables (cache once)
        self._static_cache = {
            d_static_full2abr[v]: np.asarray(self.static_vars[d_static_full2abr[v]].data)
            for v in self.dict_vars["static_vars"]
        }

    def __len__(self):
        return self.len_dataobj
    
    def _prepare_inds_for_forecast(self, lead_time_h=6):
        # Determine if this is training or validation data based on time range
        first_time = np.min(self.inds)
        last_time = np.max(self.inds)
        
        # Create a dataset identifier based on time range
        dataset_id = f"{self.name}_{first_time.astype('datetime64[D]')}_{last_time.astype('datetime64[D]')}"
        cache_file = os.path.join('utils', f'forecast_pairs_{lead_time_h}h_{dataset_id}_lenInds{(len(self.inds))}.pkl')

        # Try to load from cache first
        if os.path.exists(cache_file) and self.with_cache:
            try:
                with open(cache_file, 'rb') as f:
                    self.d_ind_pairs = pickle.load(f)
                    ind_pair_key = f"{lead_time_h}h_forecast"
                    if ind_pair_key in self.d_ind_pairs:
                        self.len_dataobj = len(self.d_ind_pairs[ind_pair_key])
                        return
            except (EOFError, pickle.UnpicklingError):
                # Handle corrupt cache file
                print(f"Warning: Cache file {cache_file} is corrupted. Recreating...")
        
        # If cache doesn't exist or is invalid, compute from scratch
        # Compute the indices
        x_t1 = self.inds
        x_t0 = x_t1 - np.timedelta64(lead_time_h, 'h')
        y_t = x_t1 + np.timedelta64(lead_time_h, 'h')
        l_pairs = []
        for i in range(len(x_t1)):
            if x_t0[i] in self.ds.time and x_t1[i] in self.ds.time and y_t[i] in self.ds.time:
                l_pairs.append((x_t0[i], x_t1[i], y_t[i]))
        pairs = tuple(l_pairs)
        ind_pair_key = f"{lead_time_h}h_forecast"
        self.d_ind_pairs[ind_pair_key] = pairs
        self.len_dataobj = len(pairs)
        
        # Save to cache for future use - only from rank 0
        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0
        
        if is_rank_zero and self.with_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.d_ind_pairs, f)
            print(f'Saved forecast paired indices cache to {cache_file}')
    
    def __getitem__(self, idx):
        if self.str_task == '6h-forecast':
            return self._get_forecast(idx, lead_time_h=6)
        elif self.str_task == 'forecast' and self.lead_time_h != 0:
            return self._get_forecast(idx, lead_time_h=self.lead_time_h)
        else:
            raise ValueError(f"Invalid task: {self.str_task}")

    def _get_forecast(self, idx, lead_time_h: int = 6):
        """
        Fast version that uses positional indexing (isel) and minimises the
        number of xarray reads.
        """
        ind_key = f"{lead_time_h}h_forecast"
        if ind_key not in self.d_ind_pairs:
            raise ValueError(f"Invalid lead time: {lead_time_h}")

        x_ind0, x_ind1, y_ind = self.d_ind_pairs[ind_key][idx]

        d_t_idx = dict()
        if hasattr(self, '_time2idx_extended'):
            for var in self._time2idx_extended.keys():
                _time2idx = self._time2idx_extended[var]
                try: 
                    i0 = _time2idx[int(x_ind0.astype("datetime64[ns]").astype(int))]
                    i1 = _time2idx[int(x_ind1.astype("datetime64[ns]").astype(int))]
                    iy = _time2idx[int(y_ind.astype("datetime64[ns]").astype(int))]
                except KeyError as e:
                    missing_ts = np.datetime_as_string(e.args[0], unit="s")
                    raise KeyError(
                        f"Timestamp {missing_ts} not found in extended dataset for variable {var} after sub-setting."
                    ) from None
                d_t_idx[var] = [i0, i1, iy]
        try:
            i0 = self._time2idx[int(x_ind0.astype("datetime64[ns]").astype(int))]
            i1 = self._time2idx[int(x_ind1.astype("datetime64[ns]").astype(int))]
            iy = self._time2idx[int(y_ind.astype("datetime64[ns]").astype(int))]
        except KeyError as e:
            missing_ts = np.datetime_as_string(e.args[0], unit="s")
            raise KeyError(
                f"Timestamp {missing_ts} not found in dataset.time after sub-setting."
            ) from None
        t_idx = [i0, i1, iy]  
                                

        # surface variables - batch load all vars at once 
        surf_vars_list = list(self.dict_vars["surf_vars"])
        surf_abr_list = [d_srf_full2abr[v] for v in surf_vars_list]
        surf_abr_list_without_co2 = [abr for abr in surf_abr_list if abr != 'co2']
        
        if hasattr(self, '_time2idx_extended'):
            t_idx_surf = {abr: d_t_idx[d_srf_abr2full[abr]] if d_srf_abr2full[abr] in self._time2idx_extended else t_idx for abr in surf_abr_list_without_co2}
        else:
            t_idx_surf = {abr: t_idx for abr in surf_abr_list_without_co2}
        
        # Load all surface variables in one operation
        surf_data = np.stack([
            np.asarray(self.surf_vars[abr].isel(time=t_idx_surf[abr]).data)
            for abr in surf_abr_list_without_co2
        ])  # shape: (N_vars, 3, H, W)

        # Split into x and y dictionaries
        x_srf = {
            abr: surf_data[i, :2]  # (2, H, W)
            for i, abr in enumerate(surf_abr_list_without_co2) 
        }
        y_srf = {
            abr: surf_data[i, 2:]   # (1, H, W)
            for i, abr in enumerate(surf_abr_list_without_co2) 
        }

        # CO₂ (optional)
        if d_srf_abr2full["co2"] in self.dict_vars["surf_vars"]:
            # lazily create the CO₂ DataArray for the three timesteps
            ds_co2 = self.co2_mapper.getitem([x_ind0, x_ind1, y_ind],
                                                lat=self.lat, lon=self.lon)
            x_srf["co2"] = np.stack(
                (ds_co2.sel(time=x_ind0).values,
                    ds_co2.sel(time=x_ind1).values), axis=-3
            )                                                                     # (2,H,W)
            y_srf["co2"] = ds_co2.sel(time=[y_ind]).values                           # (1, H,W)

        x_static = self._static_cache
        y_static = self._static_cache

        # atmospheric variables - batch load all vars at once 
        atmos_vars_list = list(self.dict_vars["atmos_vars"])
        atmos_abr_list = [d_atmos_full2abr[v] for v in atmos_vars_list]
        
        # Load all atmospheric variables in one operation
        atmos_data = np.stack([
            np.asarray(self.atmos_vars[abr].isel(time=t_idx).data)
            for abr in atmos_abr_list
        ])  # shape: (N_vars, 3, L, H, W)

        # Split into x and y dictionaries
        x_atmos = {
            abr: atmos_data[i, :2]  # (2, L, H, W)
            for i, abr in enumerate(atmos_abr_list)
        }
        y_atmos = {
            abr: atmos_data[i, 2:]   # (1, L, H, W)
            for i, abr in enumerate(atmos_abr_list)
        }
        
        atmos_vars_output = [atmos_abr_list]  # Wrap in list to avoid collation
        surf_vars_output = [surf_abr_list]   # Wrap in list to avoid collation
            
        return {
            'name': self.name,
            "x_srf": x_srf,
            "x_static": x_static,
            "x_atmos": x_atmos,
            "y_srf": y_srf,
            "y_static": y_static,
            "y_atmos": y_atmos,
            "x_time": str(x_ind1),
            "y_time": str(y_ind),
            "lat": self.lat,
            "lon": self.lon,
            "atmos_levels": self.atmos_levels,
            "locations": self.locations,
            "scales": self.scales,
            "grid_resolution": self.grid_resolution,
            "is_global_observation": self.is_global_observation,
            "atmos_vars_output": atmos_vars_output,
            "surf_vars_output": surf_vars_output,
            "lead_time_seconds": timedelta(hours=self.lead_time_h).total_seconds(),
        }

class MaskDataset(Dataset):
    def __init__(self, dataset_obj, d_str_task=None, **kwargs):
        """
        A subclass of WeatherBench2Raw that will do random Masking based on str_task.
        It inherits all properties and methods from WeatherBench2Raw.
        Args:
            d_str_task (dict): The task(s) to perform and their respective probabilities. Sum must not go beyond 1.0, 
            {'spatial-unmask':0.3, 'vertical-unmask':0.3, 'variable-unmask':0.3} would mean 30% chance of 
            spatial unmasking, 30% chance of vertical unmasking, 30% chance of variable unmasking, and
            10% chance of no masking.
            prob_mask_var (float): Probability of masking a variable [0,1].
            prob_mask_spatial (float): Probability of masking spatial locations [0,1].
            prob_mask_vertical (float): Probability of masking vertical levels [0,1].
        """
        self.dataset_obj = dataset_obj
        self.len = len(dataset_obj)  # Length of the dataset
        self.d_str_task = d_str_task
        if self.d_str_task is None:
            self.d_str_task = {'spatial-unmask': 0.33, 'vertical-unmask': 0.33, 'variable-unmask': 0.33} # 0.01 chance of no masking
        else:
            if not isinstance(self.d_str_task, dict):
                raise ValueError("d_str_task must be a dictionary with task names as keys and their probabilities as values.")
            for k in self.d_str_task.keys():
                if k not in ['spatial-unmask', 'vertical-unmask', 'variable-unmask']:
                    raise ValueError(f"Invalid task name {k} in d_str_task. Valid tasks are: 'spatial-unmask', 'vertical-unmask', 'variable-unmask'.")
            if not sum(self.d_str_task.values()) <= 1.0:
                raise ValueError("The sum of probabilities in d_str_task cannot exceed 1.0.")
        self.prob_mask_var = kwargs.get('prob_mask_var', 0.2)  # Probability of masking a variable
        self.prob_mask_spatial = kwargs.get('prob_mask_spatial', 0.5)  # Probability of masking spatial locations
        self.prob_mask_vertical = kwargs.get('prob_mask_vertical', 0.3)  # Probability of masking vertical levels
        self.patch_size = kwargs.get('tokenization_patch_size', 4)  # Size of the patches to mask
        self.prng = None
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.len
        
    def _init_prng(self):
        """
        Initializes a separate pseudo-random number generator (PRNG) for each worker.
        
        This method ensures that each worker in a multi-worker data loading setup has its own
        PRNG instance, initialized with a unique seed. The seed is derived from `torch.initial_seed()`,
        which is specific to each worker. This approach guarantees reproducibility and prevents
        workers from sharing the same random number sequence.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = torch.initial_seed() % 2**32
        else:
            # In worker process
            seed = torch.initial_seed() % 2**32
        self.prng = np.random.RandomState(seed)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing masked data based on the specified str_masking_task.
        """
        d_sample = self.dataset_obj.__getitem__(idx)  # Call the parent class method to ensure all properties are set
        if self.prng is None:
            self._init_prng()
        ## select a task based on rand value btw 0 and 1, and the probabilities in d_str_task.
        rand_val = self.prng.rand()
        cumulative_prob = 0.0
        selected_task = None
        for task, prob in self.d_str_task.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                selected_task = task
                break
        if selected_task == 'spatial-unmask':
            d_sample = self._get_spatial_unmask(d_sample)
        elif selected_task == 'vertical-unmask':
            d_sample = self._get_vertical_unmask(d_sample)
        elif selected_task == 'variable-unmask':
            d_sample = self._get_variable_unmask(d_sample)
        else:
            pass # No masking applied, return the sample as is
        return d_sample
        
    def _get_spatial_unmask(self, d_sample ):
        """
        Applies spatial masking to surface & atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable.
        prob_mask_spatial: The amount of masking being applied across lat/lon for a variable that will be masked.
        """
        patch_size = self.patch_size
        if d_sample['name'] in ['station_npz', 'weather5k'] or d_sample['name'].startswith('station'):
            patch_size = 1  # no spatial masking for station data or weather5k data
        for k in d_sample['x_srf'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_srf'][k].flags.writeable:
                    d_sample['x_srf'][k] = d_sample['x_srf'][k].copy()
                # Mask the variable. Mask entire patches, assuming patch size of patch_size x patch_size.
                patched_res = np.array(np.asarray(d_sample['x_srf'][k].shape[1:])/patch_size).astype(int) # shape [lat/patch_size, lon/patch_size]
                mask = self.prng.rand(*patched_res) < self.prob_mask_spatial
                mask = mask.repeat(patch_size, axis=0).repeat(patch_size, axis=1) # shape [lat, lon]
                d_sample['x_srf'][k][:,mask] = np.nan #broadcast to time dim as well (dim 0)
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_atmos'][k].flags.writeable:
                    d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
                # Mask the variable. Mask entire patches, assuming patch size of patch_size x patch_size.
                patched_res = np.concatenate([[d_sample['x_atmos'][k].shape[1]], np.asarray(d_sample['x_atmos'][k].shape[2:])/patch_size]).astype(int) #get [atmos-levels, lat/patch_size, lon/patch_size] shape
                mask = self.prng.rand(*patched_res) < self.prob_mask_spatial
                # rescale mask to match the spatial dimensions of the variable
                mask = mask.repeat(patch_size, axis=1).repeat(patch_size, axis=2) #extend back to [atmos-levels, lat, lon] 
                d_sample['x_atmos'][k][:,mask] = np.nan # broadcast to time dim as well (dim 0)
        return d_sample
                
    def _get_vertical_unmask(self, d_sample):
        """
        Applies vertical masking to the atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable.
        prob_mask_vertical: chances of masking being applied to a given atmospheric level.
        """
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_atmos'][k].flags.writeable:
                    d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
                    
                # pick how many vertical levels to mask
                num_levels = d_sample['x_atmos'][k].shape[1]
                mask = self.prng.rand(num_levels) < self.prob_mask_vertical
                # apply the mask to the variable
                d_sample['x_atmos'][k][:, mask, :, :] = np.nan
        return d_sample
    
    def _get_variable_unmask(self, d_sample):
        """
        Applies variable masking to the surface & atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable as a whole.
        """
        for k in d_sample['x_srf'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_srf'][k].flags.writeable:
                    d_sample['x_srf'][k] = d_sample['x_srf'][k].copy()
                    
                # Mask the variable
                d_sample['x_srf'][k][:] = np.nan
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_atmos'][k].flags.writeable:
                    d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
                
                
                # Mask the variable
                d_sample['x_atmos'][k][:] = np.nan
        return d_sample
    
class MaskDatasetV2(Dataset):
    def __init__(self, dataset_obj, d_str_task=None, **kwargs):
        """
        A subclass of WeatherBench2Raw that will do random Masking based on str_task.
        It inherits all properties and methods from WeatherBench2Raw.
        Different than MaskDataset, this class will not mask a vertical level with prob_mask_var * prob_mask_vertical, but rather prob_mask_vertical.
        Args:
            d_str_task (dict): The task(s) to perform and their respective probabilities. Sum must not go beyond 1.0, 
            {'spatial-unmask':0.3, 'vertical-unmask':0.3, 'variable-unmask':0.3} would mean 30% chance of 
            spatial unmasking, 30% chance of vertical unmasking, 30% chance of variable unmasking, and
            10% chance of no masking.
            prob_mask_var (float): Probability of masking a variable [0,1].
            prob_mask_spatial (float): Probability of masking spatial locations [0,1].
            prob_mask_vertical (float): Probability of masking vertical levels [0,1].
        """
        self.dataset_obj = dataset_obj
        self.len = len(dataset_obj)  # Length of the dataset
        self.d_str_task = d_str_task
        if self.d_str_task is None:
            self.d_str_task = {'spatial-unmask': 0.33, 'vertical-unmask': 0.33, 'variable-unmask': 0.33} # 0.01 chance of no masking
        else:
            if not isinstance(self.d_str_task, dict):
                raise ValueError("d_str_task must be a dictionary with task names as keys and their probabilities as values.")
            for k in self.d_str_task.keys():
                if k not in ['spatial-unmask', 'vertical-unmask', 'variable-unmask']:
                    raise ValueError(f"Invalid task name {k} in d_str_task. Valid tasks are: 'spatial-unmask', 'vertical-unmask', 'variable-unmask'.")
            if not sum(self.d_str_task.values()) <= 1.0:
                raise ValueError("The sum of probabilities in d_str_task cannot exceed 1.0.")
        self.prob_mask_var = kwargs.get('prob_mask_var', 0.2)  # Probability of masking a variable
        self.prob_mask_spatial = kwargs.get('prob_mask_spatial', 0.5)  # Probability of masking spatial locations
        self.prob_mask_vertical = kwargs.get('prob_mask_vertical', 0.3)  # Probability of masking vertical levels
        self.patch_size = kwargs.get('tokenization_patch_size', 4)  # Size of the patches to mask
        self.prng = None
        self.return_untouched_sample = kwargs.get('return_untouched_sample', False)  # Whether to return the untouched sample along with the masked one
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.len
        
    def _init_prng(self):
        """
        Initializes a separate pseudo-random number generator (PRNG) for each worker.
        
        This method ensures that each worker in a multi-worker data loading setup has its own
        PRNG instance, initialized with a unique seed. The seed is derived from `torch.initial_seed()`,
        which is specific to each worker. This approach guarantees reproducibility and prevents
        workers from sharing the same random number sequence.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = torch.initial_seed() % 2**32
        else:
            # In worker process
            seed = torch.initial_seed() % 2**32
        self.prng = np.random.RandomState(seed)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing masked data based on the specified str_masking_task.
        """
        d_sample = self.dataset_obj.__getitem__(idx)  # Call the parent class method to ensure all properties are set
        if self.return_untouched_sample:
            d_sample_untouched = deepcopy(d_sample)
        if self.prng is None:
            self._init_prng()
        ## select a task based on rand value btw 0 and 1, and the probabilities in d_str_task.
        rand_val = self.prng.rand()
        cumulative_prob = 0.0
        selected_task = None
        for task, prob in self.d_str_task.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                selected_task = task
                break
        if selected_task == 'spatial-unmask':
            d_sample = self._get_spatial_unmask(d_sample)
        elif selected_task == 'vertical-unmask':
            d_sample = self._get_vertical_unmask(d_sample)
        elif selected_task == 'variable-unmask':
            d_sample = self._get_variable_unmask(d_sample)
        else:
            pass # No masking applied, return the sample as is
        if self.return_untouched_sample:
            return [d_sample, d_sample_untouched]
        return d_sample
    
    def get_rollout_target(self, current_time, lead_steps: int = 1, step_hours: int = None):
        return self.dataset_obj.get_rollout_target(
            current_time=current_time,
            lead_steps=lead_steps,
            step_hours=step_hours,
        )
        
    def _get_spatial_unmask(self, d_sample ):
        """
        Applies spatial masking to surface & atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable.
        prob_mask_spatial: The amount of masking being applied across lat/lon for a variable that will be masked.
        """
        patch_size = self.patch_size
        if d_sample['name'] in ['station_npz', 'weather5k'] or d_sample['name'].startswith('station'):
            patch_size = 1  # no spatial masking for station data or weather5k data
        for k in d_sample['x_srf'].keys():
            # Create writable copy if array is read-only
            if not d_sample['x_srf'][k].flags.writeable:
                d_sample['x_srf'][k] = d_sample['x_srf'][k].copy()
            # Mask the variable. Mask entire patches, assuming patch size of patch_size x patch_size.
            patched_res = np.array(np.asarray(d_sample['x_srf'][k].shape[1:])/patch_size).astype(int) # shape [lat/patch_size, lon/patch_size]
            mask = self.prng.rand(*patched_res) < self.prob_mask_spatial
            mask = mask.repeat(patch_size, axis=0).repeat(patch_size, axis=1) # shape [lat, lon]
            d_sample['x_srf'][k][:,mask] = np.nan #broadcast to time dim as well (dim 0)
        for k in d_sample['x_atmos'].keys():
            # Create writable copy if array is read-only
            if not d_sample['x_atmos'][k].flags.writeable:
                d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
            # Mask the variable. Mask entire patches, assuming patch size of patch_size x patch_size.
            patched_res = np.concatenate([[d_sample['x_atmos'][k].shape[1]], np.asarray(d_sample['x_atmos'][k].shape[2:])/patch_size]).astype(int) #get [atmos-levels, lat/patch_size, lon/patch_size] shape
            mask = self.prng.rand(*patched_res) < self.prob_mask_spatial
            # rescale mask to match the spatial dimensions of the variable
            mask = mask.repeat(patch_size, axis=1).repeat(patch_size, axis=2) #extend back to [atmos-levels, lat, lon] 
            d_sample['x_atmos'][k][:,mask] = np.nan # broadcast to time dim as well (dim 0)
        return d_sample
                
    def _get_vertical_unmask(self, d_sample):
        """
        Applies vertical masking to the atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable.
        prob_mask_vertical: chances of masking being applied to a given atmospheric level.
        """
        for k in d_sample['x_atmos'].keys():
            # Create writable copy if array is read-only
            if not d_sample['x_atmos'][k].flags.writeable:
                d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
                
            # pick how many vertical levels to mask
            num_levels = d_sample['x_atmos'][k].shape[1]
            mask = self.prng.rand(num_levels) < self.prob_mask_vertical
            # apply the mask to the variable
            d_sample['x_atmos'][k][:, mask, :, :] = np.nan
        return d_sample
    
    def _get_variable_unmask(self, d_sample):
        """
        Applies variable masking to the surface & atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable as a whole.
        """
        for k in d_sample['x_srf'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_srf'][k].flags.writeable:
                    d_sample['x_srf'][k] = d_sample['x_srf'][k].copy()
                    
                # Mask the variable
                d_sample['x_srf'][k][:] = np.nan
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Create writable copy if array is read-only
                if not d_sample['x_atmos'][k].flags.writeable:
                    d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
                # Mask the variable
                d_sample['x_atmos'][k][:] = np.nan
        return d_sample
    
class MaskDatasetV3(MaskDatasetV2):
    def __init__(self, dataset_obj, d_str_task=None, **kwargs):
        """
        A subclass of WeatherBench2Raw that will do random Masking based on str_task.
        It inherits all properties and methods from WeatherBench2Raw.
        Different than MaskDataset, this class will not mask a vertical level with prob_mask_var * prob_mask_vertical, but rather prob_mask_vertical.
        Args:
            d_str_task (dict): The task(s) to perform and their respective probabilities. Sum must not go beyond 1.0, 
            {'spatial-unmask':0.3, 'vertical-unmask':0.3, 'variable-unmask':0.3} would mean 30% chance of 
            spatial unmasking, 30% chance of vertical unmasking, 30% chance of variable unmasking, and
            10% chance of no masking.
            prob_mask_var (float): Probability of masking a variable [0,1].
            prob_mask_spatial (float): Probability of masking spatial locations [0,1].
            prob_mask_vertical (float): Probability of masking vertical levels [0,1].
            contiguous_spatial_mask_ratio (float): Ratio for contiguous spatial masking to apply before creating another contiguous block, default is 0.20.
        """
        super().__init__(dataset_obj, d_str_task=d_str_task, **kwargs)
        self.contiguous_spatial_mask_ratio = kwargs.get('contiguous_spatial_mask_ratio', 0.20)
        
    def create_contiguous_mask(self, lat_size, lon_size, patch_size):
        """Create mask with contiguous rectangular regions"""
        mask = np.zeros((lat_size, lon_size), dtype=bool)
        
        # Calculate number of regions and their coverage
        total_pixels = lat_size * lon_size
        target_masked_pixels = int(total_pixels * self.prob_mask_spatial)
        
        # Early return if no masking needed
        if target_masked_pixels == 0:
            return mask
        
        # Number of full-sized regions
        pixels_per_region = int(total_pixels * self.contiguous_spatial_mask_ratio)
        num_full_regions = target_masked_pixels // pixels_per_region
        remainder_pixels = target_masked_pixels % pixels_per_region
        
        # Create full-sized regions
        num_full_regions_determined = 0
        while num_full_regions_determined < num_full_regions:
            # Random latitude size for this region (ensure integer and valid range)
            lat_region_max = max(patch_size, min(lat_size, int(pixels_per_region / patch_size)))
            lat_region = self.prng.randint(patch_size, lat_region_max + 1)
            # Calculate longitude size to achieve target pixel coverage
            lon_region = min(pixels_per_region // lat_region, lon_size)
            if lon_region < 1:
                lon_region = 1
                lat_region = min(pixels_per_region, lat_size)
            
            # Ensure region fits within grid bounds
            lat_region = min(lat_region, lat_size)
            lon_region = min(lon_region, lon_size)
            
            # Random center position (ensure valid bounds)
            if lat_region > 0 and lon_region > 0:
                # Ensure we have valid ranges for randint
                lat_min = max(lat_region // 2, 0)
                lat_max = min(lat_size - lat_region // 2, lat_size)
                lon_min = max(lon_region // 2, 0)
                lon_max = min(lon_size - lon_region // 2, lon_size)
                
                if lat_min < lat_max and lon_min < lon_max:
                    lat_center = self.prng.randint(lat_min, lat_max)
                    lon_center = self.prng.randint(lon_min, lon_max)
                    
                    # Apply mask for this region
                    lat_start = max(0, lat_center - lat_region // 2)
                    lat_end = min(lat_size, lat_start + lat_region)
                    lon_start = max(0, lon_center - lon_region // 2)
                    lon_end = min(lon_size, lon_start + lon_region)
                    
                    mask[lat_start:lat_end, lon_start:lon_end] = True
                    num_full_regions_determined += 1
        
        # Create remainder region if needed
        if remainder_pixels > 0:
            # Random latitude size for remainder region (ensure valid range)
            lat_region_max = max(1, min(lat_size, remainder_pixels))
            lat_region = self.prng.randint(1, lat_region_max + 1)
            # Calculate longitude size for remainder
            lon_region = max(1, min(remainder_pixels // lat_region, lon_size))
            if lon_region < 1:
                lon_region = 1
                lat_region = min(remainder_pixels, lat_size)
            
            # Ensure region fits within grid bounds
            lat_region = min(lat_region, lat_size)
            lon_region = min(lon_region, lon_size)
            
            # Random center position (ensure valid bounds)
            if lat_region > 0 and lon_region > 0:
                # Ensure we have valid ranges for randint
                lat_min = max(lat_region // 2, 0)
                lat_max = min(lat_size - lat_region // 2, lat_size)
                lon_min = max(lon_region // 2, 0)
                lon_max = min(lon_size - lon_region // 2, lon_size)
                
                if lat_min < lat_max and lon_min < lon_max:
                    lat_center = self.prng.randint(lat_min, lat_max)
                    lon_center = self.prng.randint(lon_min, lon_max)
                    
                    # Apply mask for remainder region
                    lat_start = max(0, lat_center - lat_region // 2)
                    lat_end = min(lat_size, lat_start + lat_region)
                    lon_start = max(0, lon_center - lon_region // 2)
                    lon_end = min(lon_size, lon_start + lon_region)
                    
                    mask[lat_start:lat_end, lon_start:lon_end] = True
        
        return mask

    def _get_spatial_unmask(self, d_sample):
        """
        Applies spatial masking to surface & atmospheric variables in the sample.
        Creates large contiguous rectangular regions based on contiguous_spatial_mask_ratio.
        
        If prob_mask_spatial=0.5 and contiguous_spatial_mask_ratio=0.2, creates:
        - 2 regions covering ~20% each  
        - 1 region covering ~10% (remainder)
        """
        patch_size = self.patch_size
        if d_sample['name'] in ['station_npz', 'weather5k'] or d_sample['name'].startswith('station'):
            patch_size = 1  # no spatial masking for station data or weather5k data
        lat_size = None
        mask = None
        for k in d_sample['x_srf'].keys():
            # Create writable copy if array is read-only
            if not d_sample['x_srf'][k].flags.writeable:
                d_sample['x_srf'][k] = d_sample['x_srf'][k].copy()
            
            # Get spatial dimensions (lat, lon)
            lat_size, lon_size = d_sample['x_srf'][k].shape[1:]
            mask = self.create_contiguous_mask(lat_size, lon_size, patch_size)
            d_sample['x_srf'][k][:, mask] = np.nan
            
        for k in d_sample['x_atmos'].keys():
            # Create writable copy if array is read-only
            if not d_sample['x_atmos'][k].flags.writeable:
                d_sample['x_atmos'][k] = d_sample['x_atmos'][k].copy()
                
            # Get spatial dimensions (lat, lon) - assume shape is [time, levels, lat, lon]
            if lat_size is None:
                lat_size, lon_size = d_sample['x_atmos'][k].shape[2:]
                # Create a single mask for each level
            mask = self.create_contiguous_mask(lat_size, lon_size, patch_size)

            d_sample['x_atmos'][k][:, :, mask] = np.nan
        return d_sample

class CMIP6ClimaXDataset(Dataset):
    """
    A PyTorch IterableDataset for loading and iterating over CMIP6 climate data stored in .npz files.
    This Class is only for forcastign and assumes timesteps are always 6 hours.
    Args:
        path (str): Path to the directory containing the .npz files.
        start_idx (int, optional): Starting index for the files to be loaded. 
        end_idx (int, optional): Ending index for the files to be loaded.
        surf_vars (list[str], optional): List of surface variables to be loaded (based on the names in original dataset).
        static_vars (list[str], optional): List of static variables to be loaded (based on the names in original dataset).
        atmos_vars (list[str], optional): List of atmospheric variables to be loaded (based on the names in original dataset).
        variable_name_mapping (dict, optional): A dictionary to map the variables names from original dataset to a uniform naming convension (dict(zip(source_var_naming, ESFM_var_naming))).
        atmos_levels (list[int], optional): List of atmospheric levels to be loaded.
        lat (int, optional): Number of latitude points.
        lon (int, optional): Number of longitude points.
        shuffle (bool, optional): Whether to shuffle the file list.
    Methods:
        find_first_times_key(keys):
            Finds the first key in the provided keys that ends with '_times'.
        convert_to_strtime(time):
            Converts a time object to a string representation.
        iterate_over_single_chunk(path):
            Iterates over a single .npz file and yields data dictionaries.
        __iter__():
            Iterates over the dataset, yielding data dictionaries.
    """

    def __init__(
        self,
        path,
        name='cmip6',
        start_idx: int = 0,
        end_idx: int = None,
        surf_vars: list[str] = ['psl'],
        static_vars: list[str] = [],
        atmos_vars: list[str] = ['va', 'ta', 'ua', 'zg'],
        variable_name_mapping: dict = None,
        atmos_levels: list[int] = [50, 850, 500, 600, 250, 700, 925],
        shuffle: bool = False,
        sample_per_chunk=None,
        str_task: str='6h-forecast',
        wb2_path: str=None,
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name
        data_dir = os.path.join(path, 'train/*.npz')
        self.lon = np.load(os.path.join(path, 'lon.npy'))
        self.lat = np.flip(np.sort(np.load(os.path.join(path, 'lat.npy'))))
        self.file_list = natsorted(glob.glob(data_dir))
        assert len(self.file_list), f'There is no .npz files under: {data_dir}'
        
        self.str_task = str_task
        if str_task == "6h-forecast":
            self.lead_time_h = 6
        else:
            raise NotImplementedError(f"Task {str_task} is not implemented for CMIP6 dataset yet.")
        if variable_name_mapping is None:
            # variable_name_mapping is used to map the name of the variables
            self.variable_name_mapping = {k: v for k, v in zip(
                ['ta', 'ua', 'va', 'zg', 'hus', 'tas', 'uas', 'vas', 'psl'],
                ['t', 'u', 'v', 'z', 'q', '2t', '10u', '10v', 'msl']
            )}
        else:
            self.variable_name_mapping = variable_name_mapping
        
        # Calculate actual number of samples per file by loading each file
        self.samples_per_file = []
        self.cumulative_samples = [0]  # Cumulative sum for indexing
        
        for file_path in self.file_list:
            data = mmnpz.load(file_path)
            # Get the number of timesteps from the 'times' key or first variable
            num_timesteps = data['times'].shape[0] if 'times' in data else next(iter(data.values())).shape[0]
            
            # Adjust for forecast task (need at least 3 timesteps: x_(n-2), x_(n-1) -> x_n)
            if self.str_task == '6h-forecast':
                num_samples = max(0, num_timesteps - 2)
            else:
                num_samples = num_timesteps
                
            self.samples_per_file.append(num_samples)
            self.cumulative_samples.append(self.cumulative_samples[-1] + num_samples)
        
        self.num_samples = self.cumulative_samples[-1]
        if end_idx is None:
            end_idx = self.num_samples - 1
        assert start_idx < self.num_samples, f"start_idx {start_idx} is out of range for total samples {self.num_samples}"
        assert end_idx <= self.num_samples, f"end_idx {end_idx} is out of range for total samples {self.num_samples}"
        assert start_idx < end_idx, f"start_idx {start_idx} should be smaller than end_idx {end_idx}"

        self.indices = list(range(start_idx, end_idx))
        self.surf_vars = surf_vars
        self.static_vars = static_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels
        if isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)
        self.shuffle = shuffle
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution

        # mapping the cmip6 variable names to ERA5 names
        self.locations, self.scales = load_normalization_stats(
            path, variable_name_mapping=variable_name_mapping
        )
        if wb2_path and static_vars:
            wb2 = xr.open_zarr(wb2_path) # TODO: check if some CMIP6 data come from different earth models, in this case it doesn't make sense to use era5 static variables

            # Interpolate static variables to match the latitude and longitude of the CMIP6 dataset
            self.static_vars = {
                var: wb2[d_static_abr2full[var]]
                .interp(latitude=self.lat, longitude=self.lon, method='nearest').values
                for var in static_vars
            }
        else:
            self.static_vars = {}

    def __len__(self):
        return len(self.indices)
    
    def find_first_times_key(self, keys):
        for key in keys:
            if key.endswith('_times'):
                return key
        return None

    def convert_to_strtime(self, time):
        if isinstance(time, (cftime.DatetimeNoLeap, datetime)):
            return time.strftime('%Y-%m-%dT%H:%M:%S.%f')
        return time 
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if self.str_task == '6h-forecast':
            return self._get_forecast(actual_idx)
        else:
            raise ValueError(f"Invalid task: {self.str_task}")

    def _get_forecast(self, idx):
        # Find which file this index belongs to using binary search
        file_idx = np.searchsorted(self.cumulative_samples[1:], idx, side='right')
        # Calculate the local index within the file
        i = idx - self.cumulative_samples[file_idx]
        try:
            data = mmnpz.load(self.file_list[file_idx])
        except Exception as e:
            print(f"Dataset: {self.name}: Error loading file.")
            raise RuntimeError(f"Dataset: {self.name}: Error loading file {self.file_list[file_idx]}: {e}")
        times = data['times']
        try: 
            _ = times[i+2]
        except Exception as e:
            print(f"Dataset: {self.name}: Error accessing time index {i+2} in file {self.file_list[file_idx]}.")
            raise RuntimeError(f"Dataset: {self.name}: Error accessing time index {i+2} in file {self.file_list[file_idx]}: {e}")   
        data_dict = {
            'name': self.name,
            'x_time': times[i+1].astype('datetime64[s]').item().strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'y_time': times[i+2].astype('datetime64[s]').item().strftime('%Y-%m-%dT%H:%M:%S.%f'), 
            'x_srf': {}, 
            'x_atmos': {}, 
            'x_static': {},
            'y_srf': {}, 
            'y_atmos': {},
            'y_static': {}, 
            'lat': self.lat.copy(), 
            'lon': self.lon.copy(),
            'atmos_levels': self.atmos_levels,
            'locations': self.locations,
            'scales': self.scales,
            'grid_resolution': self.grid_resolution,
            'is_global_observation': self.is_global_observation,
            'lead_time_seconds': timedelta(hours=self.lead_time_h).total_seconds(),
        }

        for var in self.atmos_vars:
            # use variable_name_mapping to rename the variable if provided
            var_name = self.variable_name_mapping.get(var, var)

            x_data, y_data = [], []
            for level in self.atmos_levels:
                key = f'{var}_{level}'

                if key in data.keys():
                    x_data.append(data[key][i:i+2])
                    y_data.append(data[key][i+2:i+3])
                else:
                    x_data.append(np.full((2, 1, len(self.lat), len(self.lon)), np.nan))
                    y_data.append(np.full((1, 1, len(self.lat), len(self.lon)), np.nan))

            data_dict['x_atmos'][var_name] = np.concatenate(x_data, axis=1)
            data_dict['y_atmos'][var_name] = np.concatenate(y_data, axis=1)

        for var in self.surf_vars:
            # use variable_name_mapping to rename the variable if provided
            var_name = self.variable_name_mapping.get(var, var)
            data_dict['x_srf'][var_name] = data[var][i:i+2]
            data_dict['y_srf'][var_name] = data[var][i+2:i+3]

        # Assumed CMIP6 data doesn't have any static variables, and all static variables are from WB2 dataset
        data_dict['x_static'] = self.static_vars
        data_dict['y_static'] = self.static_vars

        if not data_dict['x_static']  and not data_dict['x_srf']:
            # Add a dummy tensor of nans if there are no 2d features
            B, C, H, W = next(iter(data_dict['x_atmos'].values())).shape
            data_dict['x_srf']['<N/A>'] = torch.full((B, H, W), float('nan')) 

        return data_dict

class StatefulMultiDatasetLoader(DataLoader):
    """
    A StatefulDataLoader that cycles through multiple datasets with different batch sizes and samplers.
    Switches between datasets every n steps and supports checkpointing through state_dict and load_state_dict.
    
    Args:
        datasets (List[Dataset]): List of PyTorch datasets to load from
        batch_sizes (List[int]): Batch size for each dataset
        data_source_ratios (List[int]): Shows how many time each dataset will through samples, compare to others.
        samplers (List[Optional[torch.utils.data.Sampler]]): Sampler for each dataset
        switch_steps (int): Number of steps before switching to the next dataset
        collate_fns (List[Optional[Callable]]): Collate function for each dataset
        num_workers (int): Number of workers for all dataloaders
        pin_memory (bool): Whether to pin memory for all dataloaders
        drop_last (bool): Whether to drop the last incomplete batch for all dataloaders
        timeout (float): Timeout value for all dataloaders
        worker_init_fn (Optional[Callable]): Worker init function for all dataloaders
        multiprocessing_context (Optional[str]): Multiprocessing context for all dataloaders
        prefetch_factor (int): Prefetch factor for all dataloaders
        persistent_workers (bool): Whether to use persistent workers for all dataloaders
        snapshot_every_n_steps (int): How often to snapshot the state for checkpointing
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        data_source_ratios: List[int] = None,
        samplers: List[Optional[torch.utils.data.Sampler]] = None,
        switch_steps: int = 1,
        collate_fns: List[Optional[Callable]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        snapshot_every_n_steps: Optional[int] = 1,
    ):
        # Validate inputs
        assert len(datasets) > 0, "Must provide at least one dataset"
        assert len(batch_sizes) == len(datasets), "Must provide a batch size for each dataset"

        self.data_source_ratios = [1] * len(datasets) if data_source_ratios is None else data_source_ratios
             
        if samplers is None:
            samplers = [None] * len(datasets)
        assert len(samplers) == len(datasets), "Must provide a sampler for each dataset"
        
        if collate_fns is None:
            collate_fns = [None] * len(datasets)
        assert len(collate_fns) == len(datasets), "Must provide a collate_fn for each dataset"
        
        # Initialize the parent DataLoader class
        super().__init__(
            dataset=None,  # No single dataset; this is a multi-dataset loader
            batch_size=1,  # Placeholder; actual batch sizes are handled internally
            shuffle=False,  # Shuffle is handled internally
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
        
        # Store parameters specific to multi-dataset functionality
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.switch_steps = switch_steps
        self.collate_fns = collate_fns
        self.common_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "pin_memory_device": pin_memory_device,
            "in_order": in_order,
            "snapshot_every_n_steps": snapshot_every_n_steps,
        }
        
        # Create individual stateful dataloaders
        self.dataloaders = []
        for i, dataset in enumerate(self.datasets):
            dataloader = StatefulDataLoader(
                dataset=dataset,
                batch_size=self.batch_sizes[i],
                shuffle=False,  # We'll use the provided samplers
                sampler=self.samplers[i],
                collate_fn=self.collate_fns[i],
                **self.common_kwargs
            )
            self.dataloaders.append(dataloader)
        
        # Length is the sum of all dataset lengths (in batches)
        self.lengths = [len(dataloader) for dataloader in self.dataloaders]
        self._length = sum(self.lengths)
        
        # Keep track of current dataset index and step counter
        self.current_dataset_idx = 0
        self.step_counter = 0
        self.current_iterators = None
        
    def __iter__(self):
        # But we're going to override with our custom multi-dataset iteration logic
        # Create iterators for each dataloader if they don't exist
        if self.current_iterators is None:
            self.current_iterators = [iter(dataloader) for dataloader in self.dataloaders]
        
        # Initialize counters if this is a fresh iterator
        dataset_idx = self.current_dataset_idx
        step_counter = self.step_counter
        
        # Loop until all dataloaders are exhausted
        while True:
            try:
                # Switch dataset if necessary
                if step_counter % self.switch_steps == 0:
                    dataset_idx = (dataset_idx + 1) % len(self.datasets)
                    self.current_dataset_idx = dataset_idx
                
                for _ in range(self.data_source_ratios[dataset_idx]):
                    # Try to get the next batch from the current dataset
                    batch = next(self.current_iterators[dataset_idx])
                    yield batch
                
                # Increment step counter
                step_counter += 1
                self.step_counter = step_counter
                
            except StopIteration:
                # Replace the exhausted iterator
                self.current_iterators[dataset_idx] = iter(self.dataloaders[dataset_idx])
                
                # If all datasets are exhausted, break the loop
                if all(not bool(len(list(itertools.islice(iter(dl), 1)))) for dl in self.dataloaders):
                    break
    
    def __len__(self):
        return self._length
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the state of all dataloaders and the multi-dataset cycling state.
        """
        # Get the state of each individual dataloader
        dataloader_states = [dataloader.state_dict() for dataloader in self.dataloaders]
        
        # Combine with the multi-dataset specific state
        state = {
            "dataloader_states": dataloader_states,
            "current_dataset_idx": self.current_dataset_idx,
            "step_counter": self.step_counter,
        }
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the state from a previously saved state_dict.
        """
        if state_dict == {}:
            return
            
        # Load the state for each individual dataloader
        dataloader_states = state_dict.get("dataloader_states", [])
        for i, dataloader_state in enumerate(dataloader_states):
            if i < len(self.dataloaders):
                self.dataloaders[i].load_state_dict(dataloader_state)
        
        # Set the multi-dataset specific state
        self.current_dataset_idx = state_dict.get("current_dataset_idx", 0)
        self.step_counter = state_dict.get("step_counter", 0)
        
        # Reset the iterators to force rebuilding them on next __iter__ call
        self.current_iterators = None
        
        # Tell the parent StatefulDataLoader to reset its iterator
        self._iterator = None
    
    def get_current_dataset_index(self):
        """Return the index of the currently active dataset"""
        return self.current_dataset_idx
    
    def reset(self):
        """Reset the dataloader to start from the first dataset"""
        self.current_dataset_idx = 0
        self.step_counter = 0
        self.current_iterators = None
        self._iterator = None


class _UnifiedDataset(Dataset):
    """
    Internal unified dataset that wraps multiple datasets with metadata.
    Each sample knows which dataset it belongs to.
    """
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        self.total_length = sum(self.dataset_lengths)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Returns (dataset_idx, sample) tuple
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for total length {self.total_length}")
        
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        return dataset_idx, self.datasets[dataset_idx][local_idx]


class _MultiDatasetBatchSampler:
    """
    Custom batch sampler that switches between datasets according to 
    switch_steps and data_source_ratios, yielding batches of indices.
    
    Supports distributed training by partitioning data across GPUs.
    """
    def __init__(
        self,
        dataset_lengths: List[int],
        batch_sizes: List[int],
        data_source_ratios: List[int],
        switch_steps: int,
        drop_last: bool,
        shuffle: bool = False,
        seed: int = 0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.dataset_lengths = dataset_lengths
        self.batch_sizes = batch_sizes
        self.data_source_ratios = data_source_ratios
        self.switch_steps = switch_steps
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Distributed training parameters
        if rank is None or world_size is None:
            # Auto-detect distributed environment
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
        else:
            self.rank = rank
            self.world_size = world_size
        
        # Calculate cumulative dataset offsets
        self.cumulative_lengths = np.cumsum([0] + dataset_lengths)
        
        # Pre-compute index pools for each dataset
        self._reset_index_pools()
        
    def _reset_index_pools(self):
        """Reset and optionally shuffle indices for each dataset with distributed partitioning"""
        self.index_pools = []
        self.pool_positions = [0] * len(self.dataset_lengths)
        
        rng = np.random.RandomState(self.seed + self.epoch)
        
        for i, length in enumerate(self.dataset_lengths):
            # Generate all indices for this dataset
            indices = np.arange(length) + self.cumulative_lengths[i]
            
            if self.shuffle:
                rng.shuffle(indices)
            
            # Partition indices for distributed training
            # Each rank gets a subset of indices
            if self.world_size > 1:
                # Calculate indices per rank (with padding if necessary)
                num_samples_per_rank = int(np.ceil(len(indices) / self.world_size))
                total_size = num_samples_per_rank * self.world_size
                
                # Pad with repetition if necessary to make it evenly divisible
                if len(indices) < total_size:
                    padding = total_size - len(indices)
                    # Repeat from the beginning to pad
                    indices = np.concatenate([indices, indices[:padding]])
                
                # Subsample for this rank
                indices = indices[self.rank:total_size:self.world_size]
            
            self.index_pools.append(indices.tolist())
    
    def _get_batch_indices(self, dataset_idx: int) -> Optional[List[int]]:
        """Get a batch of indices from a specific dataset"""
        batch_size = self.batch_sizes[dataset_idx]
        pool = self.index_pools[dataset_idx]
        pos = self.pool_positions[dataset_idx]
        
        # Check if we have enough samples
        if pos >= len(pool):
            return None
        
        end_pos = min(pos + batch_size, len(pool))
        batch_indices = pool[pos:end_pos]
        
        # Handle drop_last
        if self.drop_last and len(batch_indices) < batch_size:
            # Mark this dataset as exhausted by advancing position to end
            self.pool_positions[dataset_idx] = len(pool)
            return None
        
        self.pool_positions[dataset_idx] = end_pos
        return batch_indices
    
    def __iter__(self):
        """Generate batches according to switching logic"""
        self._reset_index_pools()
        
        dataset_idx = 0
        step_counter = 0
        
        # Continue until all datasets are exhausted
        while True:
            # Switch dataset if necessary
            if step_counter > 0 and step_counter % self.switch_steps == 0:
                dataset_idx = (dataset_idx + 1) % len(self.dataset_lengths)
            
            # Yield batches according to data_source_ratios
            batches_yielded = 0
            for _ in range(self.data_source_ratios[dataset_idx]):
                batch_indices = self._get_batch_indices(dataset_idx)
                
                if batch_indices is not None:
                    yield batch_indices
                    batches_yielded += 1
                else:
                    # Current dataset exhausted, check if all are exhausted
                    all_exhausted = all(
                        self.pool_positions[i] >= len(self.index_pools[i])
                        for i in range(len(self.dataset_lengths))
                    )
                    if all_exhausted:
                        return
                    # Skip to next dataset
                    break
            
            step_counter += 1
    
    def __len__(self):
        """Estimate total number of batches for this rank in distributed training"""
        total_batches = 0
        for i, length in enumerate(self.dataset_lengths):
            # Account for distributed partitioning
            if self.world_size > 1:
                # Each rank gets a subset of the data
                num_samples_per_rank = int(np.ceil(length / self.world_size))
                rank_length = num_samples_per_rank
            else:
                rank_length = length
            
            num_batches = rank_length // self.batch_sizes[i]
            if not self.drop_last and rank_length % self.batch_sizes[i] != 0:
                num_batches += 1
            total_batches += num_batches * self.data_source_ratios[i]
        return total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling"""
        self.epoch = epoch


def _multi_collate_fn(batch, collate_fns):
    """
    Custom collate function that handles samples from different datasets.
    Groups samples by dataset_idx and applies appropriate collate_fn.
    """
    # Group samples by dataset
    dataset_groups = {}
    for dataset_idx, sample in batch:
        if dataset_idx not in dataset_groups:
            dataset_groups[dataset_idx] = []
        dataset_groups[dataset_idx].append(sample)
    
    # Since batches should be from same dataset, we should have only one group
    assert len(dataset_groups) == 1, "Batch contains samples from multiple datasets!"
    
    dataset_idx = list(dataset_groups.keys())[0]
    samples = dataset_groups[dataset_idx]
    
    # Apply appropriate collate function
    collate_fn = collate_fns[dataset_idx]
    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    
    return collate_fn(samples)


class StatefulMultiDatasetLoader2(StatefulDataLoader):
    """
    Optimized multi-dataset loader that uses a single unified DataLoader internally.
    This avoids creating multiple dataloaders and reduces memory overhead.
    
    Key improvements over StatefulMultiDatasetLoader:
    - Uses single DataLoader with unified dataset wrapper
    - Custom batch sampler handles dataset switching logic
    - Leverages DataLoader's built-in multiprocessing for parallel data loading
    - Supports proper state checkpointing through StatefulDataLoader
    - Much more memory efficient for many datasets
    - Built-in distributed training support (automatically detects DDP environment)
    
    Args:
        datasets (List[Dataset]): List of PyTorch datasets to load from
        batch_sizes (List[int]): Batch size for each dataset
        data_source_ratios (List[int]): How many batches to yield from each dataset per cycle
        switch_steps (int): Number of steps before switching to the next dataset
        collate_fns (List[Optional[Callable]]): Collate function for each dataset
        shuffle (bool): Whether to shuffle indices within each dataset
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory
        drop_last (bool): Whether to drop the last incomplete batch
        timeout (float): Timeout value for data loading
        worker_init_fn (Optional[Callable]): Worker init function
        multiprocessing_context (Optional[str]): Multiprocessing context
        prefetch_factor (Optional[int]): Prefetch factor
        persistent_workers (bool): Whether to use persistent workers
        pin_memory_device (str): Device for pinning memory
        seed (int): Random seed for shuffling
        rank (Optional[int]): Rank for distributed training (auto-detected if None)
        world_size (Optional[int]): World size for distributed training (auto-detected if None)
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        data_source_ratios: Optional[List[int]] = None,
        switch_steps: int = 1,
        collate_fns: Optional[List[Optional[Callable]]] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        seed: int = 0,
        rank: Optional[int] = None,  # For distributed training
        world_size: Optional[int] = None,  # For distributed training
    ):
        # Validate inputs
        assert len(datasets) > 0, "Must provide at least one dataset"
        assert len(batch_sizes) == len(datasets), "Must provide a batch size for each dataset"

        if data_source_ratios is None:
            data_source_ratios = [1] * len(datasets)
        assert len(data_source_ratios) == len(datasets), "Must provide a ratio for each dataset"
        
        if collate_fns is None:
            collate_fns = [None] * len(datasets)
        assert len(collate_fns) == len(datasets), "Must provide a collate_fn for each dataset"
        
        # Store parameters
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.data_source_ratios = data_source_ratios
        self.collate_fns = collate_fns
        self.switch_steps = switch_steps
        self.seed = seed
        
        # Create unified dataset
        unified_dataset = _UnifiedDataset(datasets)
        
        # Create custom batch sampler with distributed support
        dataset_lengths = [len(d) for d in datasets]
        batch_sampler = _MultiDatasetBatchSampler(
            dataset_lengths=dataset_lengths,
            batch_sizes=batch_sizes,
            data_source_ratios=data_source_ratios,
            switch_steps=switch_steps,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        
        # Create custom collate function
        def collate_wrapper(batch):
            return _multi_collate_fn(batch, self.collate_fns)
        
        # Initialize parent StatefulDataLoader
        super().__init__(
            dataset=unified_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_wrapper,
            num_workers=num_workers,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
        
        # Store reference to batch sampler for convenience methods
        self._custom_batch_sampler = batch_sampler
        
        # Log distributed setup info
        if batch_sampler.world_size > 1:
            import logging
            logging.info(
                f"StatefulMultiDatasetLoader2: Distributed training enabled - "
                f"Rank {batch_sampler.rank}/{batch_sampler.world_size}, "
                f"Total batches per rank: {len(batch_sampler)}"
            )
        
        # Initialize snapshot state to avoid Lightning CombinedLoader issues
        # This prevents "Please call `iter(combined_loader)` first" errors
        try:
            if hasattr(self, '_snapshot'):
                self._snapshot = {}
        except Exception:
            pass  # Ignore if attribute doesn't exist or can't be set
    
    def __len__(self):
        """Return the total number of batches"""
        return len(self._custom_batch_sampler)
    
    def state_dict(self):
        """
        Return state dict for checkpointing.
        Overrides parent to ensure Lightning compatibility.
        """
        try:
            # Try to get parent's state_dict
            state = super().state_dict()
        except (RuntimeError, AssertionError):
            # If parent fails (e.g., before iteration), return minimal valid state
            state = {}
        
        # Ensure _snapshot key exists for Lightning CombinedLoader compatibility
        if '_snapshot' not in state:
            state['_snapshot'] = {}
        
        # Add our custom state
        state['_custom_batch_sampler_epoch'] = getattr(self._custom_batch_sampler, 'epoch', 0)
        
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load state from checkpoint.
        Overrides parent to handle our custom state.
        """
        if not state_dict:
            return
        
        # Load custom state
        if '_custom_batch_sampler_epoch' in state_dict:
            self._custom_batch_sampler.epoch = state_dict['_custom_batch_sampler_epoch']
        
        # Try to load parent state (may fail if not iterated yet)
        try:
            super().load_state_dict(state_dict)
        except (RuntimeError, AssertionError, KeyError):
            # Ignore errors from parent if we haven't iterated yet
            pass
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling"""
        if hasattr(self._custom_batch_sampler, 'set_epoch'):
            self._custom_batch_sampler.set_epoch(epoch)
    
    def get_current_dataset_index(self):
        """
        Get the current dataset index (approximation based on state).
        Note: This is approximate since the actual switching happens in the batch sampler.
        """
        if hasattr(self._custom_batch_sampler, '_current_dataset_idx'):
            return self._custom_batch_sampler._current_dataset_idx
        return 0
    
    def reset(self):
        """Reset the dataloader by creating a new iterator"""
        # Reset batch sampler
        if hasattr(self._custom_batch_sampler, '_reset_index_pools'):
            self._custom_batch_sampler._reset_index_pools()


class StationGridNPZ(torch.utils.data.Dataset):
    """
    Simplified dataset that ONLY uses `inds` to pick samples.

    NPZ fields (same as before):
      - data:        (T, V, H, W)
      - timestamps:  array-like (datetime64 / str / python datetime)
      - variables:   (V,) bytes or str
      - lon_grid, lat_grid
      - locations:   (V,)
      - scales:      (V,)

    Args:
      path: npz file path
      inds: sequence of center timestamps (datetime-like). For "6h-forecast" we need
            (t-Δ, t, t+Δ) all present; otherwise we need (t-Δ, t).
      surf_vars: variables to include (by original names in NPZ). Defaults to all.
      variable_name_mapping: optional rename mapping {orig_name -> new_name}.
      lead_time_h: Δ (hours)
      str_task: "6h-forecast" or others (others use two-frame (t-Δ, t) -> (t))
    """

    def __init__(
        self,
        path: str,
        inds: Sequence[Union[str, np.datetime64, object]],
        name: str = "station_npz",
        surf_vars: Optional[List[str]] = None,
        variable_name_mapping: Optional[Dict[str, str]] = None,
        lead_time_h: int = 6,
        str_task: str = "6h-forecast",
        is_global_observation: bool = True,
        grid_resolution: float = 2.0,
        atmos_levels = np.asarray([1000,], dtype=np.int32),
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.str_task = str_task
        self.lead_time_h = int(lead_time_h)
        self.is_global_observation = is_global_observation
        self.grid_resolution = float(grid_resolution)
        self.d_ind_pairs = {}
        self._pair_key = None
        
        # Set atmos_levels attribute (may be None for station data without atmospheric levels)
        self.atmos_levels = atmos_levels
        if self.atmos_levels is not None and isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)

        # Support hybrid format: separate .npy for data + .npz for metadata
        # This allows proper memory mapping of the large data array

        # Check if using hybrid format (data.npy + meta.npz)
        if path.endswith('_data.npy') or (path.endswith('.npz') and os.path.exists(path.replace('.npz', '_data.npy'))):
            # Hybrid format detected
            if path.endswith('.npz'):
                # User provided .npz path, derive .npy path
                data_path = path.replace('.npz', '_data.npy')
                meta_path = path.replace('.npz', '_meta.npz')
            else:
                # User provided _data.npy path
                data_path = path
                meta_path = path.replace('_data.npy', '_meta.npz')

            # Load data array with memory mapping (efficient!)
            self.data = np.load(data_path, mmap_mode='r')

            # Load metadata from small NPZ (fits in RAM)
            self._npz = np.load(meta_path, allow_pickle=True)
            times_raw = self._npz["timestamps"]
            vars_raw = self._npz["variables"]
            self.vars = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
            self.lon = self._npz["lon_grid"]
            self.lat = self._npz["lat_grid"]
            self._locations_all = dict(zip(self.vars, self._npz["locations"]))
            self._scales_all = dict(zip(self.vars, self._npz["scales"]))
            self.dtype = self.data.dtype

        else:
            # Original NPZ format (WARNING: cannot be properly memory-mapped!)
            self._npz = np.load(path, allow_pickle=True, mmap_mode='r')
            self.data = self._npz["data"]                     # (T, V, H, W) - loads to RAM!

            times_raw = self._npz["timestamps"]
            vars_raw = self._npz["variables"]
            self.vars = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
            self.lon = self._npz["lon_grid"]
            self.lat = self._npz["lat_grid"]
            self._locations_all = dict(zip(self.vars, self._npz["locations"]))
            self._scales_all = dict(zip(self.vars, self._npz["scales"]))
            self.dtype = self.data.dtype

        # variable selection / rename
        if surf_vars is None:
            surf_vars = self.vars
        if variable_name_mapping is None:
            variable_name_mapping = {}

        missing = [v for v in surf_vars if v not in self.vars]
        if missing:
            raise ValueError(f"Variables not found in NPZ: {missing}")

        self.surf_vars = [variable_name_mapping.get(v, v) for v in surf_vars]
        # map exposed name -> index in original V
        self._var2idx = {new: self.vars.index(orig) for orig, new in zip(surf_vars, self.surf_vars)}

        # normalize times to datetime64[h]
        def to_dt64h_array(arr: Sequence[Union[str, np.datetime64, object]]) -> np.ndarray:
            a = np.asarray(arr)
            if np.issubdtype(a.dtype, np.datetime64):
                return a.astype("datetime64[h]")
            return np.array([np.datetime64(x) for x in a], dtype="datetime64[h]")

        self._to_dt64h_array = to_dt64h_array

        # Pre-compute index triplets (t-Δ, t, t+Δ) for the forecast task
        self._prepare_inds_for_forecast(path, times_raw, inds)

        self._meta_const = dict(
            name=self.name,
            lat=self.lat,
            lon=self.lon,
            locations={vn: self._locations_all[vn] for vn in self.surf_vars},
            scales={vn: self._scales_all[vn] for vn in self.surf_vars},
            grid_resolution=self.grid_resolution,
            is_global_observation=self.is_global_observation,
            lead_time_seconds=timedelta(hours=self.lead_time_h).total_seconds(),
        )

    def _prepare_centers_for_forecast(self, npz_path: str, times_raw, inds):
        """
        Pre-compute and cache valid center indices for forecast pairs.
        Also caches the timestamps array and inds to avoid reprocessing on subsequent loads.
        Similar to WeatherBench2Raw._prepare_inds_for_forecast but for NPZ files.
        """
        # Create cache file name - need to peek at inds range without full conversion
        # Convert just first/last for cache filename
        inds_arr = np.asarray(inds)
        first_ind_raw = inds_arr[0] if len(inds_arr) > 0 else None
        last_ind_raw = inds_arr[-1] if len(inds_arr) > 0 else None

        # Quick conversion for cache key only
        if first_ind_raw is not None:
            first_time = np.datetime64(first_ind_raw, 'D')
            last_time = np.datetime64(last_ind_raw, 'D')
            dataset_id = f"{first_time}_{last_time}"
        else:
            dataset_id = "empty"

        # Use NPZ filename in cache name
        npz_basename = os.path.basename(npz_path).replace('.npz', '')
        cache_file = os.path.join('utils', f'station_centers_{npz_basename}_{self.str_task}_{self.lead_time_h}h_{dataset_id}_lenInds{len(inds)}.pkl')
        print(f'Station cache_file: {cache_file}')

        # Try to load from cache first (BEFORE expensive inds conversion!)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self._centers = cached_data['centers']
                    self._times_h = cached_data['times_h']
                    self._inds_h = cached_data.get('inds_h', None) 
                    self._len = len(self._centers)

                    if self._inds_h is None:
                        print(f'Old cache format detected, converting inds...')
                        self._inds_h = self._to_dt64h_array(inds)
                    else:
                        print(f'Loaded {self._len} centers, {len(self._times_h)} timestamps, and {len(self._inds_h)} inds from cache')
                    return
            except (EOFError, pickle.UnpicklingError, KeyError) as e:
                print(f"Warning: Cache file {cache_file} is corrupted or outdated. Recreating...")

        # If cache doesn't exist or is invalid, compute from scratch
        print(f'Computing centers from scratch (this may take a while)...')

        # Convert inds (expensive operation, only done when no cache)
        self._inds_h = self._to_dt64h_array(inds)

        # Convert timestamps (only done once, then cached)
        self._times_h = self._to_dt64h_array(times_raw)
        lead = np.timedelta64(self.lead_time_h, "h")
        # map time (as int hours since epoch) -> index
        t2i = {int(t.astype("int64")): i for i, t in enumerate(self._times_h)}

        # build centers ONLY from inds
        keys = set(t2i.keys())
        th_i = self._inds_h.astype("int64")
        lead_i = lead.astype("int64") # lead time in integer.
        # For forecast tasks (6h, 1h, etc.), we need (t-Δ), t, and (t+Δ) all present
        # The str_task format is typically "Xh-forecast" where X is the lead time
        if self.str_task == "forecast" or self.str_task.endswith("-forecast"):
            # Forecast: need (t-Δ), t, and (t+Δ) for proper input pair and target
            centers = [t2i[k] for k in th_i if k in keys and (k - lead_i) in keys and (k + lead_i) in keys]
        else:
            # Other tasks (e.g., nowcasting): only need (t-Δ) and t
            centers = [t2i[k] for k in th_i if k in keys and (k - lead_i) in keys]

        if not centers:
            raise ValueError("No valid centers from provided `inds` (missing neighbors at ±lead_time).")

        self._centers = np.array(sorted(set(centers)), dtype=int)
        self._len = len(self._centers)

        # Save to cache for future use - only from rank 0
        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0

        if is_rank_zero:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'centers': self._centers,
                    'times_h': self._times_h,
                    'inds_h': self._inds_h
                }, f)
            print(f'Saved {self._len} centers, {len(self._times_h)} timestamps, and {len(self._inds_h)} inds to cache')

    def _prepare_inds_for_forecast(self, npz_path: str, times_raw, inds):
        """
        Build and cache index triplets (t-Δ, t, t+Δ) respecting the lead time in hours.
        Stores the triplets in `self.d_ind_pairs` and sets `_centers` for compatibility.
        """
        inds_arr = np.asarray(inds)
        first_ind_raw = inds_arr[0] if len(inds_arr) > 0 else None
        last_ind_raw = inds_arr[-1] if len(inds_arr) > 0 else None

        if first_ind_raw is not None:
            first_time = np.datetime64(first_ind_raw, 'D')
            last_time = np.datetime64(last_ind_raw, 'D')
            dataset_id = f"{first_time}_{last_time}"
        else:
            dataset_id = "empty"

        npz_basename = os.path.basename(npz_path).replace('.npz', '')
        cache_file = os.path.join('utils', f'station_pairs_{npz_basename}_{self.str_task}_{self.lead_time_h}h_{dataset_id}_lenInds{len(inds)}.pkl')
        pair_key = f"{self.lead_time_h}h_forecast" if (self.str_task == "forecast" or self.str_task.endswith("-forecast")) else f"{self.lead_time_h}h_{self.str_task}"
        print(f'Station pair cache_file: {cache_file}')

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                pairs = cached_data.get('pairs')
                if pairs is not None:
                    self._times_h = cached_data.get('times_h')
                    self._inds_h = cached_data.get('inds_h', None)
                    if self._inds_h is None:
                        self._inds_h = self._to_dt64h_array(inds)
                    self.d_ind_pairs = {pair_key: pairs}
                    self._centers = np.array([p[1] for p in pairs], dtype=int)
                    self._len = len(pairs)
                    self._pair_key = pair_key
                    print(f'Loaded {self._len} forecast triplets from cache')
                    return
            except (EOFError, pickle.UnpicklingError, KeyError) as e:
                print(f"Warning: Cache file {cache_file} is corrupted or outdated. Recreating...")

        print('Computing forecast pairs from scratch (this may take a while)...')

        self._inds_h = self._to_dt64h_array(inds)
        self._times_h = self._to_dt64h_array(times_raw)
        lead = np.timedelta64(self.lead_time_h, "h")
        t2i = {int(t.astype("int64")): i for i, t in enumerate(self._times_h)}

        keys = set(t2i.keys())
        th_i = self._inds_h.astype("int64")
        lead_i = lead.astype("int64")

        pairs = []
        if self.str_task == "forecast" or self.str_task.endswith("-forecast"):
            for k in th_i:
                if k in keys and (k - lead_i) in keys and (k + lead_i) in keys:
                    i_prev = t2i[k - lead_i]
                    i_cur = t2i[k]
                    i_y = t2i[k + lead_i]
                    pairs.append((i_prev, i_cur, i_y))
        else:
            for k in th_i:
                if k in keys and (k - lead_i) in keys:
                    i_prev = t2i[k - lead_i]
                    i_cur = t2i[k]
                    pairs.append((i_prev, i_cur, i_cur))

        if not pairs:
            raise ValueError("No valid forecast pairs from provided `inds` (missing neighbors at ±lead_time).")

        pairs = tuple(pairs)
        self.d_ind_pairs = {pair_key: pairs}
        self._centers = np.array([p[1] for p in pairs], dtype=int)
        self._len = len(pairs)
        self._pair_key = pair_key

        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0
        if is_rank_zero:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'pairs': pairs,
                    'times_h': self._times_h,
                    'inds_h': self._inds_h
                }, f)
            print(f'Saved {self._len} forecast triplets to cache')

    @property
    def centers(self) -> np.ndarray:
        return self._centers
    
    @property
    def inds(self) -> np.ndarray:
        """Expose _inds_h as inds for compatibility with eval scripts."""
        return self._inds_h

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Dict]:
        pair_key = self._pair_key or (f"{self.lead_time_h}h_forecast" if (self.str_task == "forecast" or self.str_task.endswith("-forecast")) else f"{self.lead_time_h}h_{self.str_task}")
        if pair_key not in self.d_ind_pairs:
            raise ValueError(f"No cached index triplets found for key {pair_key}")

        i_prev, i_cur, i_y = self.d_ind_pairs[pair_key][idx]

        x_time = np.datetime_as_string(self._times_h[i_cur], unit="s") + ".000000"
        y_time = np.datetime_as_string(self._times_h[i_y], unit="s") + ".000000"

        H, W = self.data.shape[-2], self.data.shape[-1]
        dummy_name = "<N/A>"
        x_atmos = {dummy_name: np.full((2, 1, H, W), np.nan, dtype=self.dtype)}
        y_atmos = {dummy_name: np.full((1, 1, H, W), np.nan, dtype=self.dtype)}

        x_srf = {vn: self.data[(i_prev, i_cur), self._var2idx[vn]] for vn in self.surf_vars}
        y_srf = {vn: self.data[(i_y,), self._var2idx[vn]] for vn in self.surf_vars}
        return {
            "name": self.name,
            "x_time": x_time,
            "y_time": y_time,
            "x_srf": x_srf,
            "x_static": {},
            "x_atmos": x_atmos,
            "y_srf": y_srf,
            "y_static": {},
            "y_atmos": y_atmos,
            "lat": self.lat,
            "lon": self.lon,
            "atmos_levels": self.atmos_levels,
            "locations": self._meta_const["locations"],
            "scales": self._meta_const["scales"],
            "grid_resolution": self._meta_const["grid_resolution"],
            "is_global_observation": self._meta_const["is_global_observation"],
            'lead_time_seconds': self._meta_const['lead_time_seconds'],
        }

class WeatherBench2Multi(WeatherBench2Raw):
    def __init__(
        self, 
        name='era5',
        path: str | list[str] = '/capstor/store/cscs/ERA5/weatherbench2_original', 
        extended_path: dict = None, # key=variable full name, value=path to dataset that contains this extra variable
        extended_vars: list = None, # list of the variables (full name) from extended_dataset to include in original dataset
        stats_path: str = 'esfm/normalization_stats_1979_2021.json',
        inds = None, 
        str_task: str = 'forecast', 
        dict_vars: dict = None, 
        surf_vars: list[str] = None,
        static_vars: list[str] = None,
        atmos_vars: list[str] = None,
        atmos_levels = np.asarray([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int32),
        dict_stats: Optional[dict[str, tuple[float, float]]] = None, 
        co2_path: str = '/capstor/store/cscs/swissai/a01/hydrological_data/global_annual_CO2.csv',
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        lead_time_h: int = 6, # lead time in hours for the forecast task
        with_cache: bool = True, # whether to use caching for station pairs/centers
        **kwargs,
    ):
        '''Defining surf_vars, static_vars, atmos_vars will overwrite dict_vars. 
        variable_name_mapping is ignored since all other datasets must conform to ERA5 convention.'''
        self.name = name
        self.path = path
        self.inds = inds
        self.d_ind_pairs = {}
        self.str_task = str_task
        self.dict_vars = dict_vars
        self.atmos_levels = atmos_levels
        if isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)
        self.dict_stats = dict_stats
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution
        if isinstance(path, str):
            self.ds = xr.open_zarr(path)
            self.ds_list = None
            self.ds_time = self.ds.time
        elif isinstance(path, list):
            ds_list = [xr.open_zarr(p) for p in path]
            self.ds = None
            self.ds_list = ds_list
            self.ds_time = xr.concat([ds.time for ds in ds_list], dim="time")
            # self.ds = xr.concat(ds_list, dim="time") ## causes issues with non overalapping vars (e.g., static vars)
        else:
            raise ValueError(f'path must be str or list of str, not {type(path)}')
        self.dict_ds_extended = dict()
        self.lead_time_h = lead_time_h
        self.with_cache = with_cache
        
        lat = self.ds.latitude if self.ds is not None else self.ds_list[0].latitude
        if len(lat) == 721:
            self.lat = lat.values[:-1] ## get only 720 out of the 721 latitudes
        else:
            self.lat = lat.values
        long = self.ds.longitude if self.ds is not None else self.ds_list[0].longitude
        self.lon = long.values
        self.lead_time_h = lead_time_h
        self.lead_time_x_hist = kwargs.pop('lead_time_x_hist', lead_time_h) # lead time in hours for the reconstruction task
        
        self.locations, self.scales = load_normalization_stats(stats_path)
        
        if extended_path:
            for var in extended_vars:
                # self.dict_ds_extended[var] = xr.open_zarr(extended_path[var])
                self.dict_ds_extended[d_srf_abr2full[var]] = xr.open_zarr(extended_path[var])[d_srf_abr2full[var].replace('_log', '')].sel(latitude=self.lat, longitude=self.lon)
            # self.ds = self.ds.assign({var: self.dict_ds_extended[var] for var in extended_vars}) # remove because not lazy load
        
        if self.inds is None: 
            raise AssertionError("dataset indices not provided.")
        self.len_dataobj = len(self.inds) ## will be later overwritten.
        if self.dict_vars is None:
            self.dict_vars = {
                'surf_vars': ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"),
                'static_vars': ("land_sea_mask", "geopotential_at_surface", "soil_type"),
                'atmos_vars': ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity")
            }
        if self.ds is not None:
            self.ds = self.ds.sel(level=self.atmos_levels)
            self.ds = self.ds.sel(latitude=self.lat)
            self.ds = self.ds.sel(longitude=self.lon)
            
            if surf_vars is not None:
                self.surf_vars = dict()
                for k in surf_vars:
                    if d_srf_abr2full[k] in self.ds.data_vars:
                        self.surf_vars[k] = self.ds[d_srf_abr2full[k]]
                    elif '_log' in d_srf_abr2full[k]:
                        full_name_ = d_srf_abr2full[k].replace('_log', '')
                        if full_name_ in self.ds.data_vars:
                            self.surf_vars[k] = self.ds[full_name_]
                            print(f'Using {full_name_} instead of {d_srf_abr2full[k]} from ds for dataset.')
                    else:
                        if k != 'co2':
                            self.surf_vars[k] = self.dict_ds_extended[d_srf_abr2full[k]]

                self.dict_vars['surf_vars'] = tuple([d_srf_abr2full[k] for k in surf_vars])
            else:
                self.surf_vars = {d_srf_full2abr[var]: self.ds[var] for var in self.dict_vars['surf_vars']} ## respecting the abbreviations from Aurora implementation for dict keys
            if static_vars is not None:
                self.static_vars = {k: self.ds[d_static_abr2full[k]] for k in static_vars}
                self.dict_vars['static_vars'] = tuple([d_static_abr2full[k] for k in static_vars])
            else:
                self.static_vars = {d_static_full2abr[var]: self.ds[var] for var in self.dict_vars['static_vars']}
            if atmos_vars is not None:
                self.atmos_vars = {k: self.ds[d_atmos_abr2full[k]] for k in atmos_vars}
                self.dict_vars['atmos_vars'] = tuple([d_atmos_abr2full[k] for k in atmos_vars])
            else:
                self.atmos_vars = {d_atmos_full2abr[var]: self.ds[var] for var in self.dict_vars['atmos_vars']}
        elif self.ds_list is not None:
            self.ds_list = [ds.sel(level=self.atmos_levels).sel(latitude=self.lat).sel(longitude=self.lon) for ds in self.ds_list]
        
            if surf_vars is not None:
                self.surf_vars = dict()
                for k in surf_vars:
                    if d_srf_abr2full[k] in self.ds_list[0].data_vars:
                        self.surf_vars[k] = xr.concat([ds[d_srf_abr2full[k]] for ds in self.ds_list], dim='time')
                    elif '_log' in d_srf_abr2full[k]:
                        full_name_ = d_srf_abr2full[k].replace('_log', '')
                        if full_name_ in self.ds_list[0].data_vars:
                            self.surf_vars[k] = xr.concat([ds[full_name_] for ds in self.ds_list], dim='time')
                            print(f'Using {full_name_} instead of {d_srf_abr2full[k]} from ds for dataset.')
                    else:
                        if k != 'co2':
                            if d_srf_abr2full[k] in self.dict_ds_extended:
                                self.surf_vars[k] = self.dict_ds_extended[d_srf_abr2full[k]]
                            else:
                                raise ValueError(f'Variable {d_srf_abr2full[k]} not found in any dataset for surf_vars.')

                self.dict_vars['surf_vars'] = tuple([d_srf_abr2full[k] for k in surf_vars])
            else:
                raise ValueError('When using multiple datasets (path as list), surf_vars must be specified.')
            if static_vars is not None:
                ## static vars can exist in just one xarray, figure out which and take the first that exists.
                self.static_vars = dict()
                for k in static_vars:
                    for ds in self.ds_list:
                        if d_static_abr2full[k] in ds.data_vars:
                            self.static_vars[k] = ds[d_static_abr2full[k]]
                            break
                self.dict_vars['static_vars'] = tuple([d_static_abr2full[k] for k in static_vars])
            else:
                raise ValueError('When using multiple datasets (path as list), static_vars must be specified.')
            if atmos_vars is not None:
                self.atmos_vars = dict()
                for k in atmos_vars:
                    self.atmos_vars[k] = xr.concat([ds[d_atmos_abr2full[k]] for ds in self.ds_list if d_atmos_abr2full[k] in ds.data_vars], dim='time')
                self.dict_vars['atmos_vars'] = tuple([d_atmos_abr2full[k] for k in atmos_vars])
            else:
                raise ValueError('When using multiple datasets (path as list), atmos_vars must be specified.')
        
        
        if self.str_task == '6h-forecast':
            self._prepare_inds_for_forecast(lead_time_h=6) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)
        elif self.str_task == 'forecast' and lead_time_h != 0:
            self._prepare_inds_for_forecast(lead_time_h=lead_time_h) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)

        if 'co2' in surf_vars: # must be executed after defining self.lat and self.lon
            self.co2_mapper = ScalarCO2Mapper(co2_path = co2_path,
                                  co2_fullname = d_srf_abr2full['co2'],
                                  lead_time_h = self.lead_time_h,
                                  inds = self.inds
                                  )
        
        # timestamp → position map (built once per worker) 
        times_ns = self.ds_time.values.astype("datetime64[ns]")
        self._time2idx = {int(t): idx for idx, t in enumerate(times_ns)}

        # static variables (cache once)
        self._static_cache = {
            d_static_full2abr[v]: np.asarray(self.static_vars[d_static_full2abr[v]].data)
            for v in self.dict_vars["static_vars"]
        }


    def __len__(self):
        return self.len_dataobj
    
    def _prepare_inds_for_forecast(self, lead_time_h=6):
        # Determine if this is training or validation data based on time range
        first_time = np.min(self.inds)
        last_time = np.max(self.inds)
        
        # Create a dataset identifier based on time range
        dataset_id = f"{self.name}_{first_time.astype('datetime64[D]')}_{last_time.astype('datetime64[D]')}"
        cache_file = os.path.join('utils', f'forecast_pairs_{lead_time_h}h_{dataset_id}_lenInds{(len(self.inds))}.pkl')

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.d_ind_pairs = pickle.load(f)
                    ind_pair_key = f"{lead_time_h}h_forecast"
                    if ind_pair_key in self.d_ind_pairs:
                        self.len_dataobj = len(self.d_ind_pairs[ind_pair_key])
                        return
            except (EOFError, pickle.UnpicklingError):
                # Handle corrupt cache file
                print(f"Warning: Cache file {cache_file} is corrupted. Recreating...")
        
        # If cache doesn't exist or is invalid, compute from scratch
        # Compute the indices
        x_t1 = self.inds
        x_t0 = x_t1 - np.timedelta64(lead_time_h, 'h')
        y_t = x_t1 + np.timedelta64(lead_time_h, 'h')
        l_pairs = []
        for i in range(len(x_t1)):
            if x_t0[i] in self.ds_time and x_t1[i] in self.ds_time and y_t[i] in self.ds_time:
                l_pairs.append((x_t0[i], x_t1[i], y_t[i]))
        pairs = tuple(l_pairs)
        ind_pair_key = f"{lead_time_h}h_forecast"
        self.d_ind_pairs[ind_pair_key] = pairs
        self.len_dataobj = len(pairs)
        
        # Save to cache for future use - only from rank 0
        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0
        
        if is_rank_zero and self.with_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.d_ind_pairs, f)


class MODISDataset(WeatherBench2Raw):
    def __init__(
        self, 
        name='modis',
        path: str = 'a122/MODIS/MODIS_PWV_hourly.zarr', 
        extended_path: dict = None, # key=variable full name, value=path to dataset that contains this extra variable
        extended_vars: list = None, # list of the variables (full name) from extended_dataset to include in original dataset
        stats_path: str = '/capstor/store/cscs/swissai/a122/MODIS/normalization_stats_2014-2020.json',
        inds = None, 
        str_task: str = 'forecast', 
        dict_vars: dict = None, 
        surf_vars: list[str] = None,
        static_vars: list[str] = None,
        atmos_vars: list[str] = None,
        atmos_levels = np.asarray([1000,], dtype=np.int32),
        variable_name_mapping: dict = None,
        # dict_stats: Optional[dict[str, tuple[float, float]]] = None, 
        co2_path: str = '/capstor/store/cscs/swissai/a01/hydrological_data/global_annual_CO2.csv',
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        lead_time_h: int = 6, # lead time in hours for the forecast task
        with_cache: bool = True,
        **kwargs,
    ):
        '''Defining surf_vars, static_vars, atmos_vars will overwrite dict_vars. 
        variable_name_mapping is ignored since all other datasets must conform to ERA5 convention.'''
        self.name = name
        self.path = path
        self.inds = inds
        self.d_ind_pairs = {}
        self.str_task = str_task
        self.lead_time_h = lead_time_h
        self.with_cache = with_cache
        if variable_name_mapping is None:
            # variable_name_mapping is used to map the name of the variables
            self.variable_name_mapping = {
                'MOD05_L2_avg_IR' : 'ir_mod',
                'MOD05_L2_avg_NIR' : 'nir_mod',
                'MYD05_L2_avg_IR' : 'ir_myd',
                'MYD05_L2_avg_NIR' : 'nir_myd',
            }
        else:
            self.variable_name_mapping = variable_name_mapping
        self.dict_vars = dict_vars
        if self.dict_vars is None:
            self.dict_vars = {
                'surf_vars': None,
                'static_vars': None,
                'atmos_vars': ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity")
            }
        self.atmos_levels = atmos_levels
        if isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution
        self.ds = xr.open_zarr(path)
        self.dict_ds_extended = dict()
        
        if len(self.ds.latitude) == 721:
            self.lat = self.ds.latitude.values[:-1] ## get only 720 out of the 721 latitudes
        else:
            self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values
        
        self.lead_time_x_hist = kwargs.pop('lead_time_x_hist', lead_time_h) # lead time in hours for the reconstruction task

        self.locations, self.scales = load_normalization_stats(stats_path)
        
        if extended_path:
            for var in extended_vars:
                ## static or other vars from ERA5
                if var in d_static_abr2full.keys():
                    if isinstance(extended_path[var], list):
                        self.dict_ds_extended[d_static_abr2full[var]] = xr.open_zarr(extended_path[var][0], chunks=None)[d_static_abr2full[var]].sel(latitude=self.lat, longitude=self.lon) # static vars are static, just take the first one
                    else:
                        self.dict_ds_extended[d_static_abr2full[var]] = xr.open_zarr(extended_path[var], chunks=None)[d_static_abr2full[var]].sel(latitude=self.lat, longitude=self.lon)
                elif var in d_srf_abr2full.keys():
                    if isinstance(extended_path[var], list):
                        ds_list_ = [xr.open_zarr(p, chunks=None)[d_srf_abr2full[var].replace('_log', '')].sel(latitude=self.lat, longitude=self.lon) for p in extended_path[var]]
                        merged = _merge_time_ordered(ds_list_)
                        self.dict_ds_extended[d_srf_abr2full[var]] = _subset_time_window(merged, self.inds, max(int(self.lead_time_h), int(self.lead_time_x_hist)))
                    else:
                        self.dict_ds_extended[d_srf_abr2full[var]] = xr.open_zarr(extended_path[var], chunks=None)[d_srf_abr2full[var].replace('_log', '')].sel(latitude=self.lat, longitude=self.lon)
                elif var in d_atmos_abr2full.keys():
                    if isinstance(extended_path[var], list):
                        ds_list_ = [xr.open_zarr(p, chunks=None)[d_atmos_abr2full[var]].sel(level=self.atmos_levels, latitude=self.lat, longitude=self.lon) for p in extended_path[var]]
                        merged = _merge_time_ordered(ds_list_)
                        self.dict_ds_extended[d_atmos_abr2full[var]] = _subset_time_window(merged, self.inds, max(int(self.lead_time_h), int(self.lead_time_x_hist)))
                    else:
                        self.dict_ds_extended[d_atmos_abr2full[var]] = xr.open_zarr(extended_path[var], chunks=None)[d_atmos_abr2full[var]].sel(level=self.atmos_levels, latitude=self.lat, longitude=self.lon)
                else:
                    raise ValueError(f'Variable {var} from extended_vars not recognized in any category (surf, static, atmos).')
        
        if self.inds is None: 
            raise AssertionError("dataset indices not provided.")

        self.len_dataobj = len(self.inds) ## will be later overwritten.
        self.ds = self.ds.sel(latitude=self.lat)
        self.ds = self.ds.sel(longitude=self.lon)
        if surf_vars is not None:
            self.surf_vars = dict()
            for k in surf_vars:
                if d_srf_abr2full[k] in self.ds.data_vars:
                    self.surf_vars[k] = self.ds[d_srf_abr2full[k]]

            self.dict_vars['surf_vars'] = tuple([d_srf_abr2full[k] for k in surf_vars])
        else:
            self.surf_vars = {d_srf_full2abr[var]: self.ds[var] for var in self.dict_vars['surf_vars']} ## respecting the abbreviations from Aurora implementation for dict keys
        if static_vars is not None:
            self.static_vars = dict()
            for k in static_vars:
                if d_static_abr2full[k] in self.ds.data_vars:
                    self.static_vars[k] = self.ds[d_static_abr2full[k]]
                else:
                    self.static_vars[k] = self.dict_ds_extended[d_static_abr2full[k]]
            self.dict_vars['static_vars'] = tuple([d_static_abr2full[k] for k in static_vars])
        else:
            self.static_vars = {d_static_full2abr[var]: self.ds[var] for var in self.dict_vars['static_vars']}
        if atmos_vars == ['<N/A>']:
            self.atmos_vars = {'<N/A>': np.nan}
            self.dict_vars['atmos_vars'] = ('<N/A>',)
        elif atmos_vars is not None:
            self.atmos_vars = dict()
            for k in atmos_vars:
                if d_atmos_abr2full[k] in self.ds.data_vars:
                    self.atmos_vars[k] = self.ds[d_atmos_abr2full[k]]
                else:
                    self.atmos_vars[k] = self.dict_ds_extended[d_atmos_abr2full[k]]
            self.dict_vars['atmos_vars'] = tuple([d_atmos_abr2full[k] for k in atmos_vars])
        else:
            self.atmos_vars = {d_atmos_full2abr[var]: self.ds[var] for var in self.dict_vars['atmos_vars']}
        if self.str_task == '6h-forecast':
            self._prepare_inds_for_forecast(lead_time_h=6) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)
        elif self.str_task == 'forecast' and lead_time_h != 0:
            self._prepare_inds_for_forecast(lead_time_h=lead_time_h) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)
        
        # timestamp → position map (built once per worker) 
        times_ns = self.ds.time.values.astype("datetime64[ns]")
        self._time2idx = {int(t): idx for idx, t in enumerate(times_ns)}
        self._time2idx_extended = dict()
        for var in self.dict_vars['surf_vars']:
            if var not in self.ds.data_vars and var in self.dict_ds_extended:
                ds_ext = self.dict_ds_extended[var]
                times_ns_ext = ds_ext.time.values.astype("datetime64[ns]")
                self._time2idx_extended[var] = {int(t): idx for idx, t in enumerate(times_ns_ext)}
        for var in self.dict_vars['atmos_vars']:
            if var not in self.ds.data_vars and var in self.dict_ds_extended:
                ds_ext = self.dict_ds_extended[var]
                times_ns_ext = ds_ext.time.values.astype("datetime64[ns]")
                self._time2idx_extended[var] = {int(t): idx for idx, t in enumerate(times_ns_ext)}

        # static variables (cache once)
        self._static_cache = {
            d_static_full2abr[v]: np.asarray(self.static_vars[d_static_full2abr[v]].data)
            for v in self.dict_vars["static_vars"]
        }

    def _get_forecast(self, idx, lead_time_h: int = 6):
            """
            Fast version that uses positional indexing (isel) and minimises the
            number of xarray reads.
            """
            ind_key = f"{lead_time_h}h_forecast"
            if ind_key not in self.d_ind_pairs:
                raise ValueError(f"Invalid lead time: {lead_time_h}")

            x_ind0, x_ind1, y_ind = self.d_ind_pairs[ind_key][idx]

            d_t_idx = dict()
            if hasattr(self, '_time2idx_extended'):
                for var in self._time2idx_extended.keys():
                    _time2idx = self._time2idx_extended[var]
                    try: 
                        i0 = _time2idx[int(x_ind0.astype("datetime64[ns]").astype(int))]
                        i1 = _time2idx[int(x_ind1.astype("datetime64[ns]").astype(int))]
                        iy = _time2idx[int(y_ind.astype("datetime64[ns]").astype(int))]
                    except KeyError as e:
                        missing_ts = np.datetime_as_string(e.args[0], unit="s")
                        raise KeyError(
                            f"Timestamp {missing_ts} not found in extended dataset for variable {var} after sub-setting."
                        ) from None
                    d_t_idx[var] = [i0, i1, iy]
            try:
                i0 = self._time2idx[int(x_ind0.astype("datetime64[ns]").astype(int))]
                i1 = self._time2idx[int(x_ind1.astype("datetime64[ns]").astype(int))]
                iy = self._time2idx[int(y_ind.astype("datetime64[ns]").astype(int))]
            except KeyError as e:
                missing_ts = np.datetime_as_string(e.args[0], unit="s")
                raise KeyError(
                    f"Timestamp {missing_ts} not found in dataset.time after sub-setting."
                ) from None
            t_idx = [i0, i1, iy]  
                                  

            # surface variables - batch load all vars at once 
            surf_vars_list = list(self.dict_vars["surf_vars"])
            surf_abr_list = [d_srf_full2abr[v] for v in surf_vars_list]
            surf_abr_list_without_co2 = [abr for abr in surf_abr_list if abr != 'co2']
            # [print(f'{abr}: {len(self.surf_vars[abr])}') for abr in surf_abr_list if abr != 'co2']
            
            if hasattr(self, '_time2idx_extended'):
                t_idx_surf = {abr: d_t_idx[d_srf_abr2full[abr]] if d_srf_abr2full[abr] in self._time2idx_extended else t_idx for abr in surf_abr_list_without_co2}
            else:
                t_idx_surf = {abr: t_idx for abr in surf_abr_list_without_co2}
            
            lat = self.lat
            surf_data = np.stack([
                np.asarray(self.surf_vars[abr].isel(time=t_idx_surf[abr]).data)
                for abr in surf_abr_list_without_co2
            ])  # shape: (N_vars, 3, H, W)

            # Split into x and y dictionaries
            x_srf = {
                abr: surf_data[i, :2]  # (2, H, W)
                for i, abr in enumerate(surf_abr_list_without_co2) 
            }
            y_srf = {
                abr: surf_data[i, 2:]   # (1, H, W)
                for i, abr in enumerate(surf_abr_list_without_co2) 
            }

            # CO₂ (optional)
            if d_srf_abr2full["co2"] in self.dict_vars["surf_vars"]:
                # lazily create the CO₂ DataArray for the three timesteps
                ds_co2 = self.co2_mapper.getitem([x_ind0, x_ind1, y_ind],
                                                 lat=lat, lon=self.lon)
                x_srf["co2"] = np.stack(
                    (ds_co2.sel(time=x_ind0).values,
                     ds_co2.sel(time=x_ind1).values), axis=-3
                )                                                                     # (2,H,W)
                y_srf["co2"] = ds_co2.sel(time=[y_ind]).values                           # (1, H,W)

            x_static = self._static_cache
            y_static = self._static_cache

            # atmospheric variables - batch load all vars at once 
            atmos_vars_list = list(self.dict_vars["atmos_vars"])
            atmos_abr_list = [d_atmos_full2abr[v] for v in atmos_vars_list]
            
            if len(atmos_vars_list) == 1 and atmos_vars_list[0] == "<N/A>":
                _, H, W = x_srf[surf_abr_list_without_co2[0]].shape
                x_atmos = {"<N/A>": np.full((2, 1, H, W), np.nan) } #(2, L=1, H, W)
                y_atmos = {"<N/A>": np.full((1, 1, H, W), np.nan) } #(1, L=1, H, W)
                atmos_vars_output = [["<N/A>"]]  # Wrap in list to avoid collation
            else:
                if hasattr(self, '_time2idx_extended'):
                    t_idx_atmos = {abr: d_t_idx[d_atmos_abr2full[abr]] if d_atmos_abr2full[abr] in self._time2idx_extended else t_idx for abr in atmos_abr_list}
                else:
                    t_idx_atmos = {abr: t_idx for abr in atmos_abr_list}
                
                # Load all atmospheric variables in one operation
                atmos_data = np.stack([
                    np.asarray(self.atmos_vars[abr].isel(time=t_idx_atmos[abr]).data)
                    for abr in atmos_abr_list
                ])  # shape: (N_vars, 3, L, H, W)

                # Split into x and y dictionaries
                x_atmos = {
                    abr: atmos_data[i, :2]  # (2, L, H, W)
                    for i, abr in enumerate(atmos_abr_list)
                }
                y_atmos = {
                    abr: atmos_data[i, 2:]   # (1, L, H, W)
                    for i, abr in enumerate(atmos_abr_list)
                }
                
                atmos_vars_output = [atmos_abr_list]  # Wrap in list to avoid collation
            surf_vars_output = [surf_abr_list]   # Wrap in list to avoid collation

            return_dict = {
                'name': self.name,
                "x_srf": x_srf,
                "x_static": x_static,
                "x_atmos": x_atmos,
                "y_srf": y_srf,
                "y_static": y_static,
                "y_atmos": y_atmos,
                "x_time": str(x_ind1),
                "y_time": str(y_ind),
                "lat": lat,
                "lon": self.lon,
                "atmos_levels": self.atmos_levels,
                "locations": self.locations,
                "scales": self.scales,
                "grid_resolution": self.grid_resolution,
                "is_global_observation": self.is_global_observation,
                "atmos_vars_output": atmos_vars_output,
                "surf_vars_output": surf_vars_output,
                "lead_time_seconds": timedelta(hours=self.lead_time_h).total_seconds(),
            }

            # return_dict = ensure_contiguous(return_dict)
            
            return return_dict