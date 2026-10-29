# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from datetime import datetime, timedelta
import numpy as np
import pickle
import sys
from natsort import natsorted
import glob
import yaml
import json

from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import timedelta


# Parse custom arguments first and remove from sys.argv before other imports
import sys
import argparse

# Extract custom arguments
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--NUM_ROLLOUT_STEPS', type=int, default=56)
custom_parser.add_argument('--surf_vars_save', nargs='+', default=None)
custom_parser.add_argument('--atmos_vars_save', nargs='+', default=None)
custom_parser.add_argument('--levels_save', nargs='+', type=int, default=None)
custom_parser.add_argument('--use_inds_subset', action=argparse.BooleanOptionalAction, default=True)
custom_parser.add_argument('--test_time_intervals', type=str, default='default')
custom_parser.add_argument(
        "--save_only_masked_predictions_during_testing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For masked inference runs without region masking, save only the masked variables or pressure levels instead of all predictions. Use --no-save_only_masked_predictions_during_testing to save all variables. [default: True]",
    )
custom_args, remaining = custom_parser.parse_known_args()

NUM_ROLLOUT_STEPS = custom_args.NUM_ROLLOUT_STEPS
surf_vars_save = custom_args.surf_vars_save
atmos_vars_save = custom_args.atmos_vars_save
levels_save = custom_args.levels_save
use_inds_subset = custom_args.use_inds_subset
test_time_intervals = custom_args.test_time_intervals
save_only_masked_predictions_during_testing = custom_args.save_only_masked_predictions_during_testing
# Reconstruct sys.argv without custom arguments for the main parser
sys.argv = [sys.argv[0]] + remaining

from config import parse_args
args = parse_args()

from esfm import ESFM, Batch, Metadata, rollout

# Import custom modules
from utils import dataset, validation_utils
from utils.validation_utils import merge_prediction_files_and_copy_to_capstor

# Add environment variables for memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

if args.load_aurora_pretrain_weights and args.load_custom_pretrain_weights_str is not None:
    raise ValueError("Both --load_aurora_pretrain_weights and --load_custom_pretrain_weights_str are set. Please choose one.")

# Setup environment and paths
is_rank0 = int(os.getenv("LOCAL_RANK", "0")) == 0

if is_rank0:
    print(f'NUM_ROLLOUT_STEPS: {NUM_ROLLOUT_STEPS}')
    print(f'surf_vars_save: {surf_vars_save}')
    print(f'atmos_vars_save: {atmos_vars_save}')
    print(f'levels_save: {levels_save}')
    print(f'use_inds_subset: {use_inds_subset}')
    print(f'test_time_intervals: {test_time_intervals}')

DATA_PATH_PREFIX = os.getenv("ESFM_DATA_PATH_PREFIX", "/capstor/store/cscs/").rstrip("/") + "/"
if test_time_intervals == 'default':
    start_end_time_intervals = [
        (datetime(2023, 1, 2, 0, 0, 0), datetime(2023, 1, 8, 23, 0, 0)),
        (datetime(2023, 4, 2, 0, 0, 0), datetime(2023, 4, 8, 23, 0, 0)),
        (datetime(2023, 7, 2, 0, 0, 0), datetime(2023, 7, 8, 23, 0, 0)),
        # (datetime(2023, 7, 20, 0, 0, 0), datetime(2023, 7, 29, 23, 0, 0)), ## Doksuri 20 - 29 July 2023
        (datetime(2023, 10, 2, 0, 0, 0), datetime(2023, 10, 8, 23, 0, 0)),
        (datetime(2024, 1, 2, 0, 0, 0), datetime(2024, 1, 8, 23, 0, 0)),
        (datetime(2024, 4, 2, 0, 0, 0), datetime(2024, 4, 8, 23, 0, 0)),
        (datetime(2024, 7, 2, 0, 0, 0), datetime(2024, 7, 8, 23, 0, 0)),
        (datetime(2024, 10, 2, 0, 0, 0), datetime(2024, 10, 8, 23, 0, 0)),
    ]
else:
    raise ValueError(f"Unknown test_time_intervals: {test_time_intervals}. Choose 'default' or '2022'.")

# Build tmp_pred_fname based on test_time_intervals and use_inds_subset
if test_time_intervals == 'default':
    tmp_pred_fname_base = "tmp_predictions_rollout"
else:
    tmp_pred_fname_base = f"tmp_predictions_rollout_{test_time_intervals}"
tmp_pred_fname = f"{tmp_pred_fname_base}_subset" if use_inds_subset else tmp_pred_fname_base
inds = []
for (start_time, end_time) in start_end_time_intervals:
    inds_ = [np.datetime64(start_time + timedelta(hours=i)) for i in range(int((end_time - start_time).total_seconds() // 3600) + 1)]
    if use_inds_subset:
        inds_ = validation_utils.get_every_nth_index(inds_, n=6, shift_n_every_m=3)
    inds.extend(inds_)
    
print(f'#timesteps for inference dataset: {len(inds)}')

def get_device(use_gpu):
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"


_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"           # keeps the format in one place
def _make_batch(batch_dict, *, normalize_target: bool = True):
    """
    Convert the default-collated dictionary into (batch_x, batch_y)
    Batch objects – exactly what you were building inside train_step.
    """
    # time & level metadata
    x_time = tuple(datetime.strptime(t, _TIME_FMT) for t in batch_dict["x_time"])
    y_time = tuple(datetime.strptime(t, _TIME_FMT) for t in batch_dict["y_time"])
    atmos_levels = tuple(batch_dict["atmos_levels"][0].cpu().numpy().tolist())
    atmos_levels_output = batch_dict.get('atmos_levels_output', [None])[0]
    atmos_vars_output = batch_dict.get('atmos_vars_output', [None])[0]
    surf_vars_output = batch_dict.get('surf_vars_output', [None])[0]
    lead_time_seconds = batch_dict.get('lead_time_seconds', [timedelta(hours=6).total_seconds()])[0]
    if isinstance(lead_time_seconds, timedelta):
        lead_time_seconds = lead_time_seconds.total_seconds()
    if atmos_levels_output is None:
        atmos_levels_output = atmos_levels
    else:
        atmos_levels_output = tuple(atmos_levels_output.cpu().numpy().tolist())
    if atmos_vars_output is not None:
        atmos_vars_output = tuple([it[0] for it in atmos_vars_output])
    if surf_vars_output is not None:
        surf_vars_output = tuple([it[0] for it in surf_vars_output])
    
    # Build Batch objects
    batch_x = Batch(
        surf_vars=batch_dict["x_srf"],
        static_vars={k: v[0] for k, v in batch_dict["x_static"].items()},
        atmos_vars=batch_dict["x_atmos"],
        metadata=Metadata(
            dataset_name=batch_dict['name'][0],
            lat=batch_dict["lat"][0],
            lon=batch_dict["lon"][0],
            time=x_time,
            atmos_levels=atmos_levels,
            locations={k: v[0] for k, v in batch_dict["locations"].items()},
            scales={k: v[0] for k, v in batch_dict["scales"].items()},
            grid_resolution=batch_dict["grid_resolution"][0],
            is_global_observation=batch_dict["is_global_observation"][0],
            atmos_vars_output=atmos_vars_output,
            surf_vars_output=surf_vars_output,
            atmos_levels_output=atmos_levels_output,
            lead_time_seconds=lead_time_seconds,
        ),
    )

    batch_y = Batch(
        surf_vars=batch_dict["y_srf"],
        static_vars={k: v for k, v in batch_dict["y_static"].items()},
        atmos_vars=batch_dict["y_atmos"],
        metadata=Metadata(
            dataset_name=batch_dict['name'][0],
            lat=batch_dict["lat"][0],
            lon=batch_dict["lon"][0],
            time=y_time,
            atmos_levels=atmos_levels_output,
            locations={k: v[0] for k, v in batch_dict["locations"].items()},
            scales={k: v[0] for k, v in batch_dict["scales"].items()},
            grid_resolution=batch_dict["grid_resolution"][0],
            is_global_observation=batch_dict["is_global_observation"][0],
            atmos_vars_output=atmos_vars_output,
            surf_vars_output=surf_vars_output,
            atmos_levels_output=atmos_levels_output,
            lead_time_seconds=lead_time_seconds,
        ),
    )

    if normalize_target:
        batch_y = batch_y.normalise(surf_stats=None)

    return batch_x, batch_y

# # Define collate functions for training and validation
collate_fn_val  = lambda samples: _make_batch(torch.utils.data.default_collate(samples), normalize_target=False)

def load_data(yaml_path, datasets_type):
    # TODO: replace this function with LightningDataModule
    """Load datasets based on YAML configuration."""
    dataset_val = []
    batch_sizes = []

    for dataset_type in datasets_type:
        with open(yaml_path, 'r') as file:
            yml_file = yaml.safe_load(file)
            data_cls = getattr(dataset, yml_file[dataset_type]['class'])
            conf_infer = yml_file[dataset_type]['conf']
            path_dataset = conf_infer['path']
            if isinstance(path_dataset, list):
                conf_infer['path'] = [os.path.join(DATA_PATH_PREFIX, p) for p in path_dataset]
            else:
                conf_infer['path'] = os.path.join(DATA_PATH_PREFIX, conf_infer['path'])
            
            if dataset_type.startswith('era5') or dataset_type.startswith('station') or dataset_type.startswith('modis') or dataset_type.startswith('cosmo'):
                conf_infer['inds'] = inds
            else:
                conf_infer['start_idx'] = yml_file[dataset_type]['start_val']
                conf_infer['end_idx'] = yml_file[dataset_type]['end_val']
                if 'wb2_path' in conf_infer:
                    conf_infer['wb2_path'] = os.path.join(DATA_PATH_PREFIX, conf_infer['wb2_path'])

            batch_sizes.append(1)

            dataset_val.append(data_cls(**conf_infer))
            
    if args.devices > 1:
        dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=7200))
    if len(dataset_val) == 1:
        dataloader_val = DataLoader(
            dataset_val[0],
            sampler=DistributedSampler(dataset_val[0], shuffle=False, drop_last=False) if args.devices > 1 else None,
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            # shuffle=False,  # Ensure deterministic for validation
            drop_last=False,
            pin_memory=True,  
            persistent_workers=True,
            collate_fn=collate_fn_val  
        )
    else:
        dataloader_val = dataset.StatefulMultiDatasetLoader(
            datasets=dataset_val,
            samplers=[StatefulDistributedSampler(ds, shuffle=False, drop_last=True) for ds in dataset_val] if args.devices > 1 else None,
            batch_sizes=batch_sizes,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            collate_fns=[collate_fn_val for _ in dataset_val],
        ) 

    return dataloader_val


def gather_vars_for_model(yaml_path, datasets_type):
    # TODO: replace this function with LightningDataModule
    """Load datasets based on YAML configuration."""
    surf_vars, atmos_vars, static_vars = (), (), ()

    for dataset_type in datasets_type:
        with open(yaml_path, 'r') as file:
            yml_file = yaml.safe_load(file)

            surf_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                              for var in yml_file[dataset_type]['conf']['surf_vars']] 
                              if 'variable_name_mapping' in yml_file[dataset_type]['conf'] 
                              else yml_file[dataset_type]['conf']['surf_vars'])
            atmos_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['conf']['atmos_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['conf']['atmos_vars'])
            static_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['conf']['static_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['conf']['static_vars'])
            
    ## keep only unique var names and remove repetitions in tuple. (Note: set() op. is not deterministic.)
    surf_vars = tuple(sorted(set(surf_vars)))
    atmos_vars = tuple(sorted(set(atmos_vars)))
    static_vars = tuple(sorted(set(static_vars)))
    

    return surf_vars, atmos_vars, static_vars

def main():
    # Basic setup
    L.seed_everything(args.seed, workers=True)
    logging.info("args = %s", args)
    
    if not args.no_gpu and not torch.cuda.is_available():
        logging.info("GPU training is requested, but no GPU device available.")
        sys.exit(1)
        
    # Replace the existing dataset setup with:
    dataloader_val = load_data(
        args.dataset_config_path, args.test_data_sources, 
    )
    
    surf_vars, atmos_vars, static_vars = gather_vars_for_model(args.dataset_config_path, args.data_sources)
    
    ## setup model architecture
    encoder_act_checkpointing = args.act_checkpointing_encoder
    backbone_act_checkpointing = args.act_checkpointing_backbone
    decoder_act_checkpointing = args.act_checkpointing_decoder
    str_architecture_size = args.str_architecture_size
    if str_architecture_size == "small":
        encoder_depths = (2,6,2)
        encoder_num_heads = (4,8,16)
        decoder_depths = (2, 6, 2)
        decoder_num_heads = (16, 8, 4)
        embed_dim = 256
        num_heads = 8
        hf_pretrain_fname = 'aurora-0.25-small-pretrained.ckpt'
    elif str_architecture_size == "large":
        encoder_depths = (6, 10, 8)
        encoder_num_heads = (8, 16, 32)
        decoder_depths = (8, 10, 6)
        decoder_num_heads = (32, 16, 8)
        embed_dim= 512
        num_heads = 16
        hf_pretrain_fname = 'aurora-0.25-pretrained.ckpt'
    else:
        raise ValueError(f"Unknown architecture size: {str_architecture_size}. Choose 'small' or 'large'.")


    # Model setup
    model = ESFM(
        use_lora=False, 
        autocast=True, # Use AMP (mixed precision to fit to GPU)
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        encoder_depths=encoder_depths,
        encoder_num_heads=encoder_num_heads,
        decoder_depths=decoder_depths,
        decoder_num_heads=decoder_num_heads,
        embed_dim=embed_dim,
        num_heads=num_heads,
        drop_path=0.0,
        num_ensemble = args.num_ensemble,  # Number of ensemble members
        variable_aggregation= args.variable_aggregation,
        use_resolution_specific_patch_tokenizers = args.use_resolution_specific_patch_tokenizers,
        disable_flashattention=args.disable_flashattention,
        stabilise_level_agg=args.stabilise_level_agg,
        add_qk_norm_to_swin3d=args.add_qk_norm_to_swin3d,
        fix_qk_norm_in_swin3d=args.fix_qk_norm_in_swin3d,
        encoder_activation_checkpointing=encoder_act_checkpointing,  # Enable extensive checkpointing for large models
        absolute_time_embedding_in_minutes=args.absolute_time_embedding_in_minutes, # use absolute time embedding in minutes
        add_token_pos_embedding_in_decoder=args.add_token_pos_embedding_in_decoder, # add token positional embedding in the decoder
        num_max_ensembles=args.num_max_ensembles, # maximum number of ensembles
        ensemble_adaln_scale_bias=args.ensemble_adaln_scale_bias,
    )

    # Lightning Module
    class InferenceModule(L.LightningModule):
        def __init__(self, net, var_mask=None, plev_mask=None, mask_region=None, tmp_prediction_save_dir=None, 
                     surf_vars_save=None, atmos_vars_save=None, levels_save=None, strict_loading=True):
            super().__init__()
            self.net = net
            self.var_mask = var_mask
            self.plev_mask = plev_mask
            self.mask_region = mask_region
            self.tmp_prediction_save_dir = tmp_prediction_save_dir
            self.surf_vars_save = surf_vars_save
            self.atmos_vars_save = atmos_vars_save
            self.levels_save = levels_save
            self.printed_dropped_vars = False
            self.strict_loading = strict_loading

        def forward(self, x):
            return self.net.forward(x)

        def test_step(self, test_batch, batch_idx):
            print(f"Processing batch {batch_idx}...")
            batch_obj_x, batch_obj_y = test_batch
            
            ## If some vars in batch not in the corpus evaluated model, drop them.
            vars_drop_static = []
            for k in list(batch_obj_x.static_vars.keys()):
                if k not in static_vars:
                    vars_drop_static.append(k)
                    del batch_obj_x.static_vars[k]
            vars_drop_surf = []
            for k in list(batch_obj_x.surf_vars.keys()):
                if k not in surf_vars:
                    vars_drop_surf.append(k)
                    del batch_obj_x.surf_vars[k]
            vars_drop_atmos = []
            for k in list(batch_obj_x.atmos_vars.keys()):
                if k not in atmos_vars:
                    vars_drop_atmos.append(k)
                    del batch_obj_x.atmos_vars[k]
            if vars_drop_static or vars_drop_surf or vars_drop_atmos:
                if not self.printed_dropped_vars:
                    logging.info(f"Dropping variables not in the model from test steps: static vars {vars_drop_static}, surf vars {vars_drop_surf}, atmos vars {vars_drop_atmos}.")
                    self.printed_dropped_vars = True
            del test_batch  
            
            leadtime_seconds = batch_obj_x.metadata.lead_time_seconds
            leadtime_hours = int(leadtime_seconds // 3600)
            print(f'Init time: {batch_obj_x.metadata.time[-1]}, lead time per step: {leadtime_hours} hours.')
            
            
            if self.var_mask is not None and self.plev_mask is not None and self.mask_region is not None:
                ## Mask subset of region + plevs + vars
                atmos_levels = batch_obj_x.metadata.atmos_levels
                inds_plev_mask_binary = [h in self.plev_mask for h in atmos_levels]
                inds_plev_mask = np.where(inds_plev_mask_binary)[0]
                var_types = self.var_mask[0] if isinstance(self.var_mask[0], (list, tuple)) else [self.var_mask[0]]
                var_keys = self.var_mask[1] if isinstance(self.var_mask[1], (list, tuple)) else [self.var_mask[1]]
                lat, lon = batch_obj_x.metadata.lat.cpu().numpy(), batch_obj_x.metadata.lon.cpu().numpy()
                latlon_gridlat, latlon_gridlon = np.meshgrid(lat, lon, indexing='ij')
                latlon_grid = np.stack([latlon_gridlat.flatten(), latlon_gridlon.flatten()], axis=1)
                if isinstance(self.mask_region, str):
                    inds_linear = validation_utils.get_region_indices(latlon_grid, region_str=self.mask_region)
                    inds_lat_mask, inds_lon_mask = np.unravel_index(inds_linear, latlon_gridlat.shape)
                elif isinstance(self.mask_region, list):
                    l_inds_lat_mask, l_inds_lon_mask = [], []
                    for region_str in self.mask_region:
                        inds_linear = validation_utils.get_region_indices(latlon_grid, region_str=region_str)
                        inds_lat_mask, inds_lon_mask = np.unravel_index(inds_linear, latlon_gridlat.shape)
                        l_inds_lat_mask.append(inds_lat_mask)
                        l_inds_lon_mask.append(inds_lon_mask)
                    inds_lat_mask, inds_lon_mask = None, None
                for var_type, var_key in zip(*(var_types, var_keys)):
                    if var_type == 'surf_var':
                        # Mask region of surf_var in the batch_obj_x
                        if inds_lat_mask is not None and inds_lon_mask is not None:
                            batch_obj_x.surf_vars[var_key][:,:,inds_lat_mask, inds_lon_mask] = torch.nan
                        else:
                            for inds_lat_mask_, inds_lon_mask_ in zip(l_inds_lat_mask, l_inds_lon_mask):
                                batch_obj_x.surf_vars[var_key][:,:,inds_lat_mask_, inds_lon_mask_] = torch.nan
                    elif var_type == 'atmos_var': 
                        # Mask the asked plevs of mask region of the atmos_var in the batch_obj_x
                        if inds_lat_mask is not None and inds_lon_mask is not None:
                            # Mask each pressure level separately to avoid broadcasting issues
                            for plev_idx in inds_plev_mask:
                                batch_obj_x.atmos_vars[var_key][:, :, plev_idx, inds_lat_mask, inds_lon_mask] = torch.nan
                        else:
                            for inds_lat_mask_, inds_lon_mask_ in zip(l_inds_lat_mask, l_inds_lon_mask):
                                # Mask each pressure level separately to avoid broadcasting issues
                                for plev_idx in inds_plev_mask:
                                    batch_obj_x.atmos_vars[var_key][:, :, plev_idx, inds_lat_mask_, inds_lon_mask_] = torch.nan
                
            elif self.var_mask is not None and self.plev_mask is not None:
                # Mask subset of plevs + vars
                atmos_levels = batch_obj_x.metadata.atmos_levels
                inds_plev_mask_binary = [h in self.plev_mask for h in atmos_levels]
                var_types = self.var_mask[0] if isinstance(self.var_mask[0], (list, tuple)) else [self.var_mask[0]]
                var_keys = self.var_mask[1] if isinstance(self.var_mask[1], (list, tuple)) else [self.var_mask[1]]
                for var_type, var_key in zip(*(var_types, var_keys)):
                    if var_type == 'surf_var':
                        # Mask the complete surf_var in the batch_obj_x
                        batch_obj_x.surf_vars[var_key] *= torch.nan
                    elif var_type == 'atmos_var': 
                        # Mask the asked plevs of the atmos_var in the batch_obj_x
                        batch_obj_x.atmos_vars[var_key][:,:,inds_plev_mask_binary, :, :] = torch.nan
            else:
                ###### MASK ONLY 1 COMPONENT AT A TIME ######
                ## Block that masks certain variables during testing, if specified.
                if self.var_mask is not None and self.var_mask[0] is not None and self.var_mask[1] is not None:
                    var_types = self.var_mask[0] if isinstance(self.var_mask[0], (list, tuple)) else [self.var_mask[0]]
                    var_keys = self.var_mask[1] if isinstance(self.var_mask[1], (list, tuple)) else [self.var_mask[1]]
                    for var_type, var_key in zip(*(var_types, var_keys)):
                        if var_type == 'surf_var':
                            # Mask the surf_var in the batch_obj_x
                            batch_obj_x.surf_vars[var_key] *= torch.nan
                        elif var_type == 'atmos_var':
                            batch_obj_x.atmos_vars[var_key] *= torch.nan
                            
                if self.plev_mask is not None:
                    atmos_levels = batch_obj_x.metadata.atmos_levels
                    inds_plev_mask_binary = [h in self.plev_mask for h in atmos_levels]
                    # inds_plev_mask = [i for i, x in enumerate(inds_plev_mask_binary) if x]
                    for k in batch_obj_x.atmos_vars.keys():
                        batch_obj_x.atmos_vars[k][:,:,inds_plev_mask_binary, :, :] = torch.nan
                        
                if self.mask_region is not None:
                    lat, lon = batch_obj_x.metadata.lat.cpu().numpy(), batch_obj_x.metadata.lon.cpu().numpy()
                    latlon_gridlat, latlon_gridlon = np.meshgrid(lat, lon, indexing='ij')
                    latlon_grid = np.stack([latlon_gridlat.flatten(), latlon_gridlon.flatten()], axis=1)
                    inds_linear = validation_utils.get_region_indices(latlon_grid, region_str=self.mask_region)
                    inds_lat_mask, inds_lon_mask = np.unravel_index(inds_linear, latlon_gridlat.shape)
                    for k in batch_obj_x.surf_vars.keys():
                        batch_obj_x.surf_vars[k][:,:,inds_lat_mask, inds_lon_mask] = torch.nan
                    for k in batch_obj_x.atmos_vars.keys():
                        batch_obj_x.atmos_vars[k][:,:,:, inds_lat_mask, inds_lon_mask] = torch.nan
            
            # Helper function to safely convert to numpy
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                elif isinstance(x, (tuple, list)):
                    return np.array(x)
                return x
            
            # Setup temporary file path for incremental saving
            if self.tmp_prediction_save_dir is None:
                tmp_dir = os.path.join(args.log_dir, tmp_pred_fname)
            else:
                tmp_dir = os.path.join(args.log_dir, self.tmp_prediction_save_dir, tmp_pred_fname)
            os.makedirs(tmp_dir, exist_ok=True)
            init_time = batch_obj_x.metadata.time[-1:]
            timestamp = init_time[0].strftime('%Y%m%d_%H%M%S')
            step_dir = os.path.join(tmp_dir, f'steps_{timestamp}')
            tmp_file = os.path.join(tmp_dir, f'pred_{timestamp}.pkl')
            
            # Create directory for individual step files
            os.makedirs(step_dir, exist_ok=True)
            
            # Perform rollout inference - write each step to separate file
            rollout_gen = rollout(self.net, batch_obj_x, steps=NUM_ROLLOUT_STEPS)
            
            # Store metadata from first step
            metadata_saved = False
            
            # Iterate through rollout steps - write each to separate file
            for step_idx, batch_ens in enumerate(rollout_gen):
                curr_lead_time_hours = (step_idx + 1) * leadtime_hours  # Each step is 6 hours: 6, 12, 18, ...
                
                # Save metadata on first step
                if not metadata_saved:
                    first_surf_var = next(iter(batch_ens.surf_vars.values()))
                    num_ensembles = first_surf_var.shape[1]  # [bs, ens, ...]
                    if num_ensembles == 1:
                        num_ensembles = 0
                    
                    metadata = {
                        'init_time': init_time,
                        'latitude': to_numpy(batch_obj_y.metadata.lat),
                        'longitude': to_numpy(batch_obj_y.metadata.lon),
                        'level': to_numpy(batch_obj_y.metadata.atmos_levels),
                        'ensemble': to_numpy(num_ensembles),
                    }
                    metadata_file = os.path.join(step_dir, 'metadata.pkl')
                    with open(metadata_file, 'wb') as f:
                        pickle.dump(metadata, f)
                    metadata_saved = True
                
                # Save this step's predictions to a separate file
                step_data = {'lead_time': curr_lead_time_hours}
                
                # Filter surface variables based on surf_vars_save
                for var_name, pred_tensor in batch_ens.surf_vars.items():
                    # Skip if not in surf_vars_save list (when specified)
                    if self.surf_vars_save is not None and var_name not in self.surf_vars_save:
                        continue
                    # pred_tensor shape: [bs, ens, 1, lat, lon] -> squeeze time dim -> [bs, ens, lat, lon]
                    step_data[var_name] = pred_tensor.squeeze(2).detach().cpu().numpy()
                
                # Filter atmospheric variables based on atmos_vars_save and levels_save
                for var_name, pred_tensor in batch_ens.atmos_vars.items():
                    # Skip if not in atmos_vars_save list (when specified)
                    if self.atmos_vars_save is not None and var_name not in self.atmos_vars_save:
                        continue
                    # pred_tensor shape: [bs, ens, 1, plev, lat, lon] -> squeeze time dim -> [bs, ens, plev, lat, lon]
                    pred_np = pred_tensor.squeeze(2).detach().cpu().numpy()
                    
                    # Filter pressure levels if levels_save is specified
                    if self.levels_save is not None:
                        atmos_levels = batch_obj_x.metadata.atmos_levels
                        level_indices = [i for i, level in enumerate(atmos_levels) if level in self.levels_save]
                        pred_np = pred_np[:, :, level_indices, :, :]  # [bs, ens, plev_filtered, lat, lon]
                    
                    step_data[var_name] = pred_np
                
                # Write this step to disk (small, fast write)
                step_file = os.path.join(step_dir, f'step_{step_idx:03d}.pkl')
                with open(step_file, 'wb') as f:
                    pickle.dump(step_data, f)
                
                # Clear memory immediately
                del batch_ens, pred_tensor, step_data
                torch.cuda.empty_cache()
                
                logging.info(f"Batch {batch_idx}: Saved rollout step {step_idx + 1}/{NUM_ROLLOUT_STEPS} (lead_time={curr_lead_time_hours}h)")
            
            # Now merge all step files into final format (only reads each file once)
            logging.info(f"Batch {batch_idx}: Merging {NUM_ROLLOUT_STEPS} step files...")
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                predictions = pickle.load(f)
            
            # Update level metadata if levels_save was specified
            if self.levels_save is not None:
                predictions['level'] = np.array(self.levels_save)
            
            # Build lead_time array
            predictions['lead_time'] = np.arange(1, NUM_ROLLOUT_STEPS + 1) * leadtime_hours
            
            # Load all steps and stack them
            for step_idx in range(NUM_ROLLOUT_STEPS):
                step_file = os.path.join(step_dir, f'step_{step_idx:03d}.pkl')
                with open(step_file, 'rb') as f:
                    step_data = pickle.load(f)
                
                # Initialize arrays on first step
                if step_idx == 0:
                    for var_name, var_data in step_data.items():
                        if var_name != 'lead_time':
                            if var_data.ndim == 3:  # Surface var: [bs, ens, lat, lon]
                                predictions[var_name] = []
                            else:  # Atmos var: [bs, ens, plev, lat, lon]
                                predictions[var_name] = []
                
                # Append this step's data
                for var_name, var_data in step_data.items():
                    if var_name != 'lead_time':
                        predictions[var_name].append(var_data)
            
            # Stack all steps into final arrays
            for var_name in list(predictions.keys()):
                if var_name not in ['init_time', 'latitude', 'longitude', 'level', 'ensemble', 'lead_time']:
                    # Convert list to stacked array: [bs, lead_time, ens, ...]
                    predictions[var_name] = np.stack(predictions[var_name], axis=1)
            
            logging.info(f"Batch {batch_idx}: Merge complete, cleaning up step files...")
            
            # Post-process: Apply masking if needed and convert to full variable names
            # Determine which variables to save based on masking type
            save_only_masked = (
                save_only_masked_predictions_during_testing
                and (self.var_mask is not None or self.plev_mask is not None)
                and self.mask_region is None
            )
            
            # Build final predictions dict with full variable names
            final_predictions = {
                'init_time': predictions['init_time'],
                'latitude': predictions['latitude'],
                'longitude': predictions['longitude'],
                'level': predictions['level'],
                'lead_time': predictions['lead_time'],
                'ensemble': predictions['ensemble'],
            }
            
            if save_only_masked and self.var_mask is not None:
                # Only save the masked variables
                var_types = self.var_mask[0] if isinstance(self.var_mask[0], (list, tuple)) else [self.var_mask[0]]
                var_keys = self.var_mask[1] if isinstance(self.var_mask[1], (list, tuple)) else [self.var_mask[1]]
                
                for var_type, var_key in zip(var_types, var_keys):
                    if var_type == 'surf_var' and var_key in predictions:
                        full_name = dataset.d_srf_abr2full.get(var_key, var_key)
                        # Already in correct shape: [bs, lead_time, ens, lat, lon]
                        final_predictions[full_name] = predictions[var_key]
                    elif var_type == 'atmos_var' and var_key in predictions:
                        full_name = dataset.d_atmos_abr2full.get(var_key, var_key)
                        # Already in correct shape: [bs, lead_time, ens, plev, lat, lon]
                        final_predictions[full_name] = predictions[var_key]
            elif save_only_masked and self.plev_mask is not None:
                # Save only masked plevs for all atmos_vars
                final_predictions['level'] = predictions['level'][inds_plev_mask_binary]
                for var_name in predictions.keys():
                    if var_name not in ['init_time', 'latitude', 'longitude', 'level', 'ensemble', 'lead_time']:
                        # Check if it's an atmos var (has plev dimension)
                        if predictions[var_name].ndim == 6:  # [bs, lead_time, ens, plev, lat, lon]
                            # Filter only masked plevs: [bs, lead_time, ens, plev_masked, lat, lon]
                            pred_plev = predictions[var_name][:, :, :, inds_plev_mask_binary, ...]
                            full_name = dataset.d_atmos_abr2full.get(var_name, var_name)
                            final_predictions[full_name] = pred_plev
            else:
                # Save all variables (default behavior)
                for var_name in predictions.keys():
                    if var_name not in ['init_time', 'latitude', 'longitude', 'level', 'ensemble', 'lead_time']:
                        if predictions[var_name].ndim == 5:  # Surface var: [bs, lead_time, ens, lat, lon]
                            full_name = dataset.d_srf_abr2full.get(var_name, var_name)
                            final_predictions[full_name] = predictions[var_name]
                        else:  # Atmos var: [bs, lead_time, ens, plev, lat, lon]
                            full_name = dataset.d_atmos_abr2full.get(var_name, var_name)
                            final_predictions[full_name] = predictions[var_name]
            
            # Save final predictions
            print(f'Saving final predictions to: {tmp_file}')
            with open(tmp_file, 'wb') as f:
                pickle.dump(final_predictions, f)
            
            # Clean up individual step files
            import shutil
            shutil.rmtree(step_dir)
            logging.info(f"Batch {batch_idx}: Cleanup complete, removed {step_dir}")
            
            logging.info(f"Batch {batch_idx}: Completed rollout and saved all {NUM_ROLLOUT_STEPS} lead times")

            return final_predictions

    
    # Strategy setup
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        strategy = 'ddp'
        myrank = int(dist.get_rank())
    else:
        strategy = 'auto'
        myrank = 0

    # Configure progress bar for SLURM logs (no ANSI escape codes)
    import sys
    if not sys.stdout.isatty():
        # Non-interactive mode: print progress line-by-line without ANSI codes
        progress_bar = TQDMProgressBar(refresh_rate=10, process_position=0)
    else:
        # Interactive mode: use default progress bar
        progress_bar = TQDMProgressBar()

    # Trainer setup
    trainer = L.Trainer(
        accelerator='gpu' if not args.no_gpu else 'cpu',
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision='32-true',
        callbacks=[progress_bar],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
    )

    # Load checkpoint and run inference
    if args.ckpt_name == 'last.ckpt':
        ckpt_list = natsorted(glob.glob(os.path.join(args.log_dir, "model_ckpt-step*")))
        ## get ckpt int list:
        d_ckpt_step = {}
        for ckpt in ckpt_list:
            try:
                s = os.path.basename(ckpt).split('=')[1].split("-loss_train")[0]
                step = int(s)
                d_ckpt_step[step] = ckpt
            except Exception as e:
                logging.warning(f'Could not parse step from checkpoint filename: {ckpt}. Error: {type(e).__name__}: {e}')
                continue

        nK = int(5) # get the last ckpt that is a multiple of nK steps.
        l_ckpt_step = list(d_ckpt_step.keys())
        l_ckpt_step_nK = [step for step in l_ckpt_step if step % (nK * 1000) == 0]
        if len(l_ckpt_step_nK) == 0:
            ## no ckpt found that is a multiple of nK, use last multiple of 100 steps:
            l_ckpt_step_nK = [step for step in l_ckpt_step if step % 100 == 0]
            if len(l_ckpt_step_nK) == 0:
                raise ValueError(f'No checkpoint found that is a multiple of {nK*1000} steps. List ckpt steps: {l_ckpt_step}.')
        ckpt_step = l_ckpt_step_nK[-1] 
        ckpt_path = d_ckpt_step[ckpt_step]
    elif args.ckpt_name == 'hf_hub_download':
        from huggingface_hub import hf_hub_download
        if args.str_architecture_size == "large":
            hf_pretrain_fname = 'aurora-0.25-pretrained.ckpt'
        elif args.str_architecture_size == "small":
            hf_pretrain_fname = 'aurora-0.25-small-pretrained.ckpt'
        else:
            raise ValueError(f"Unknown architecture size: {str_architecture_size}. Choose 'small' or 'large' to use pretrained ckpts.")
        path = hf_hub_download(repo_id="microsoft/aurora", filename=hf_pretrain_fname)
        model.load_checkpoint_local(path, strict=args.strict_loading)
        ckpt_path = None
        ckpt_step = 0
    else:
        ckpt_step = -1
        ckpt_path = os.path.join(args.log_dir, args.ckpt_name)
        
    if is_rank0:
        print(f'Using checkpoint path: {ckpt_path}')
    copy_threads = []
    if args.vars_mask_during_testing is None and args.plevs_mask_during_testing is None and args.mask_regions_during_testing is None:
        # Setup model and trainer
        model_lightning = InferenceModule(model, surf_vars_save=surf_vars_save, atmos_vars_save=atmos_vars_save, levels_save=levels_save, strict_loading=args.strict_loading)
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version >= (2, 6):
            trainer.test(model_lightning, dataloader_val, ckpt_path=ckpt_path, weights_only=False)
        else:
            trainer.test(model_lightning, dataloader_val, ckpt_path=ckpt_path)
        logging.info(f"Rank {myrank}: Inference completed.")
        # Synchronize before merge - ensure all ranks finish writing predictions
        if args.devices > 1:
            logging.info(f"Rank {myrank}: Waiting at barrier before merge...")
            dist.barrier()
        if is_rank0:
            copy_thread = merge_prediction_files_and_copy_to_capstor(args.log_dir, ckpt_step, args.config, execute_copy_to_capstor=False, tmp_pred_fname=tmp_pred_fname)
            copy_threads.append(copy_thread)
    else:
        lightning_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        global_step = lightning_state_dict['global_step']
        if is_rank0:
            print(f'Loaded checkpoint at global step: {global_step}')
        state_dict = lightning_state_dict['state_dict']
        state_dict_net_rm = {k[4:]: v for k, v in state_dict.items() if k.startswith('net.')}
        model.load_state_dict(state_dict_net_rm, strict=args.strict_loading, )
        del lightning_state_dict
        if args.mask_multiple_during_testing: 
            ## mask everything passed simultaneously and do a single inference run.
            tmp_prediction_save_dir = 'mask_'
            if args.vars_mask_during_testing is not None:
                vars_mask_list = json.loads(args.vars_mask_during_testing) # Expected format: [['2t', 'surf_var'], ['10u', 'surf_var'], ..., ['t', 'atmos_var'],...]
                if len(vars_mask_list) > 0 and len(vars_mask_list[0]) == 2:
                    var_keys, var_types = zip(*vars_mask_list)
                else:
                    raise ValueError(f"Invalid vars_mask_list format: {vars_mask_list}")
                var_mask = (var_types, var_keys)
                tmp_prediction_save_dir += 'var_' + '_'.join(var_keys) + '_'
            else:
                var_mask = (None, None)
            if args.plevs_mask_during_testing is not None:
                plevs_mask = args.plevs_mask_during_testing
                tmp_prediction_save_dir += 'plev_' + '_'.join([str(h) for h in plevs_mask]) + '_'
            else:
                plevs_mask = None
            if args.mask_regions_during_testing is not None:
                l_region_str = args.mask_regions_during_testing
                tmp_prediction_save_dir += 'region_' + '_'.join(l_region_str)
            else:
                l_region_str = None
            if is_rank0:
                print(
                    f'Rank {myrank}: var_mask: {var_mask}, plevs_mask: {plevs_mask}, '
                    f'l_region_str: {l_region_str}, save_only_masked_predictions: '
                    f'{save_only_masked_predictions_during_testing}. '
                    f'Will be saving temporary files under {tmp_prediction_save_dir}'
                )
            
            model_lightning = InferenceModule(model, var_mask=var_mask, plev_mask=plevs_mask, mask_region=l_region_str, 
                                             tmp_prediction_save_dir=tmp_prediction_save_dir,
                                             surf_vars_save=surf_vars_save, atmos_vars_save=atmos_vars_save, levels_save=levels_save, strict_loading=args.strict_loading)
            trainer.test(model_lightning, dataloader_val)
            logging.info(f"Rank {myrank}: Inference for masked {var_types}, {var_keys} completed.")
            # Synchronize before merge
            if args.devices > 1:
                dist.barrier()
            if is_rank0:
                copy_thread = merge_prediction_files_and_copy_to_capstor(args.log_dir, ckpt_step, args.config, tmp_prediction_save_dir=tmp_prediction_save_dir, execute_copy_to_capstor=False, tmp_pred_fname=tmp_pred_fname)
                copy_threads.append(copy_thread)
        elif args.vars_mask_during_testing is not None:
            vars_mask_list = json.loads(args.vars_mask_during_testing) # Expected format: [['2t', 'surf_var'], ['10u', 'surf_var'], ..., ['t', 'atmos_var'],...]
            print(
                f'Rank {myrank}: vars_mask_list: {vars_mask_list}, '
                f'save_only_masked_predictions: {save_only_masked_predictions_during_testing}'
            )
            var_keys, var_types = zip(*vars_mask_list)
            zipped_group = zip(var_types, var_keys)
            for var_type, var_key in zipped_group:
                tmp_prediction_save_dir = f'mask_var_{var_type}_{var_key}'
                model_lightning = InferenceModule(model, var_mask=(var_type, var_key), tmp_prediction_save_dir=tmp_prediction_save_dir,
                                                 surf_vars_save=surf_vars_save, atmos_vars_save=atmos_vars_save, levels_save=levels_save, strict_loading=args.strict_loading)
                trainer.test(model_lightning, dataloader_val)
                logging.info(f"Rank {myrank}: Inference for masked {var_type}, {var_key} completed.")
                # Synchronize before merge
                if args.devices > 1:
                    dist.barrier()
                if is_rank0: 
                    copy_thread = merge_prediction_files_and_copy_to_capstor(args.log_dir, ckpt_step, args.config, tmp_prediction_save_dir=tmp_prediction_save_dir, execute_copy_to_capstor=False, tmp_pred_fname=tmp_pred_fname)
                    copy_threads.append(copy_thread)
        elif args.plevs_mask_during_testing is not None:
            if is_rank0:
                print(
                    f'Rank {myrank}: plevs_mask_during_testing: {args.plevs_mask_during_testing}, '
                    f'save_only_masked_predictions: {save_only_masked_predictions_during_testing}'
                )
            # plevs_mask = [int(h) for h in args.plevs_mask_during_testing.split(',')]
            plevs_mask = args.plevs_mask_during_testing
            for plev_mask in plevs_mask:
                tmp_prediction_save_dir = f'mask_plev_{plev_mask}'
                model_lightning = InferenceModule(model, plev_mask=[plev_mask], tmp_prediction_save_dir=tmp_prediction_save_dir,
                                                 surf_vars_save=surf_vars_save, atmos_vars_save=atmos_vars_save, levels_save=levels_save, strict_loading=args.strict_loading)
                trainer.test(model_lightning, dataloader_val)
                logging.info(f"Rank {myrank}: Inference for masked plev {plev_mask} completed.")
                # Synchronize before merge
                if args.devices > 1:
                    dist.barrier()
                if is_rank0:
                    copy_thread = merge_prediction_files_and_copy_to_capstor(args.log_dir, ckpt_step, args.config, tmp_prediction_save_dir=tmp_prediction_save_dir, execute_copy_to_capstor=False, tmp_pred_fname=tmp_pred_fname)
                    copy_threads.append(copy_thread)
        elif args.mask_regions_during_testing is not None:
            l_region_str = args.mask_regions_during_testing
            if is_rank0:
                print(f'Rank {myrank}: l_region_str: {l_region_str}')
            for region_str in l_region_str:
                tmp_prediction_save_dir = f'mask_region_{region_str}'
                model_lightning = InferenceModule(model, mask_region=region_str, tmp_prediction_save_dir=tmp_prediction_save_dir,
                                                 surf_vars_save=surf_vars_save, atmos_vars_save=atmos_vars_save, levels_save=levels_save, strict_loading=args.strict_loading)
                trainer.test(model_lightning, dataloader_val)
                logging.info(f"Rank {myrank}: Inference for masked region {region_str} completed.")
                # Synchronize before merge
                if args.devices > 1:
                    dist.barrier()
                if is_rank0:
                    copy_thread = merge_prediction_files_and_copy_to_capstor(args.log_dir, ckpt_step, args.config, tmp_prediction_save_dir=tmp_prediction_save_dir, execute_copy_to_capstor=False, tmp_pred_fname=tmp_pred_fname)
                    copy_threads.append(copy_thread)

    # Wait for all copy operations to complete before exiting (only rank 0 has threads)
    if is_rank0 and len(copy_threads) > 0:
        logging.info(f"Rank {myrank}: Waiting for {len(copy_threads)} copy thread(s) to complete...")
        failed_threads = []
        for i, thread in enumerate(copy_threads):
            thread.join()
            # Check if thread encountered an error
            if hasattr(thread, 'thread_exception') and thread.thread_exception['error'] is not None:
                failed_threads.append((i, thread.thread_exception['error']))
        
        if failed_threads:
            logging.error(f"Rank {myrank}: {len(failed_threads)} copy thread(s) failed!")
            for idx, error in failed_threads:
                logging.error(f"  Thread {idx} failed with: {error}")
        else:
            logging.info(f"Rank {myrank}: All copy operations completed successfully.")
    


if __name__ == "__main__":
    main()