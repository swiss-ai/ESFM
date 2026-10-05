# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import sys
import pickle
import numpy as np
import wandb
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# Import FSDP and Mixed Precision from PyTorch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from huggingface_hub import hf_hub_download

from esfm import ESFMEncoder, Batch, Metadata

## import custom modules
import yaml
from config import parse_args
from utils import dataset, logging_utils, losses
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torch.utils.data.distributed import DistributedSampler
import psutil
import time

from utils.gradient_logging import log_gradient_norms,log_weight_norms

# Check PyTorch version for weights_only parameter support
from packaging import version
PYTORCH_VERSION = version.parse(torch.__version__.split('+')[0])  # Remove any +cu suffix
SUPPORTS_WEIGHTS_ONLY = PYTORCH_VERSION.major == 2 and PYTORCH_VERSION.minor >= 6

# PyTorch 2.6+ defaults some loads to weights_only=True. Older Lightning checkpoints can
# include builtins.getattr in pickled objects, which must be explicitly allowlisted.
if SUPPORTS_WEIGHTS_ONLY:
    torch.serialization.add_safe_globals([
        getattr,
        losses.KD_Loss_activations,
    ])

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

is_rank0 = False
if int(os.getenv("LOCAL_RANK", "0")) == 0:
    is_rank0 = True

args = parse_args()
    
wandb_key = os.getenv("WANDB_KEY")
if is_rank0:  # Only attempt login on rank 0
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        if args.wnb_mode != 'disabled':
            print(f'WANDB_KEY is not set. Please set WANDB_KEY to use wandb. not logging to WANDB.')
            args.wnb_mode = 'disabled'

def get_total_gpus():
    total_gpus = int(os.getenv("WORLD_SIZE", "1"))  # Default to 1 node if not set
    return total_gpus

## setup dataset & dataloader
# Dataset scheme to use. For now, will use raw .zarr from weatherbench2, but this is to be changed later.
DATA_PATH_PREFIX = os.getenv("ESFM_DATA_PATH_PREFIX", "/capstor/store/cscs/").rstrip("/") + "/"
start_time_train = datetime(1979, 1, 1, 0, 0, 0)
end_time_train = datetime(2020, 12, 31, 23, 0, 0)
start_time_val = datetime(2021, 1, 1, 0, 0, 0)
end_time_val = datetime(2021, 12, 31, 23, 0, 0) ## last date on wb2 is 2021-12-31
inds_train = [np.datetime64(start_time_train + timedelta(hours=i)) for i in range(int((end_time_train - start_time_train).total_seconds() // 3600) + 1)]
inds_val = [np.datetime64(start_time_val + timedelta(hours=i)) for i in range(int((end_time_val - start_time_val).total_seconds() // 3600) + 1)]



def get_device(use_gpu):
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"           # keeps the format in one place
def _make_batch(batch_dict,):
    """
    Convert the default-collated dictionary into (batch_x, batch_y)
    Batch objects – exactly what you were building inside train_step.
    """

    if isinstance(batch_dict, list): ## masked dataset tuple
        d_sample, d_sample_untouched = batch_dict[0], batch_dict[1]
        batch_dict = d_sample 
        batch_dict_untouched = d_sample_untouched
    else:
        batch_dict_untouched = None

    # time & level metadata
    x_time = tuple(datetime.strptime(t, _TIME_FMT) for t in batch_dict["x_time"])
    atmos_levels = tuple(batch_dict["atmos_levels"][0].cpu().numpy().tolist())

    # Build Batch objects
    batch_x = Batch(
        surf_vars=batch_dict["x_srf"],
        static_vars={k: v[0] for k, v in batch_dict["x_static"].items()},
        atmos_vars=batch_dict["x_atmos"],
        metadata=Metadata(
            lat=batch_dict["lat"][0],
            lon=batch_dict["lon"][0],
            time=x_time,
            atmos_levels=atmos_levels,
            locations={k: v[0] for k, v in batch_dict["locations"].items()},
            scales={k: v[0] for k, v in batch_dict["scales"].items()},
            grid_resolution=batch_dict["grid_resolution"][0],
            is_global_observation=batch_dict["is_global_observation"][0],
        ),
    )
    
    if batch_dict_untouched is not None:
        batch_x_untouched = Batch(
            surf_vars=batch_dict_untouched["x_srf"],
            static_vars={k: v[0] for k, v in batch_dict_untouched["x_static"].items()},
            atmos_vars=batch_dict_untouched["x_atmos"],
            metadata=Metadata(
                lat=batch_dict_untouched["lat"][0],
                lon=batch_dict_untouched["lon"][0],
                time=x_time,
                atmos_levels=atmos_levels,
                locations={k: v[0] for k, v in batch_dict_untouched["locations"].items()},
                scales={k: v[0] for k, v in batch_dict_untouched["scales"].items()},
                grid_resolution=batch_dict_untouched["grid_resolution"][0],
                is_global_observation=batch_dict_untouched["is_global_observation"][0],
            ),
        )
        return batch_x, batch_x_untouched

    return batch_x

# # Define collate functions for training and validation
collate_fn_train = lambda samples: _make_batch(torch.utils.data.default_collate(samples)) 
collate_fn_val   = lambda samples: _make_batch(torch.utils.data.default_collate(samples))

def load_data(yaml_path, datasets_type, yaml_masking_path):
    # TODO: replace this function with LightningDataModule
    """Load datasets based on YAML configuration."""
    surf_vars, atmos_vars, static_vars = (), (), ()
    dataset_train, dataset_val = [], []
    batch_sizes = []

    with open(yaml_masking_path, 'r') as file:
        yml_file_masking = yaml.safe_load(file)
        mask_config_type = args.mask_config_type
        if mask_config_type is None:
            mask_data_cls = None
            mask_data_params = None
        else:
            mask_config = yml_file_masking[mask_config_type] 
            mask_data_cls = getattr(dataset, mask_config['class'])
            mask_data_params = mask_config['conf']
            
    for dataset_type in datasets_type:
        with open(yaml_path, 'r') as file:
            yml_file = yaml.safe_load(file)
            data_cls = getattr(dataset, yml_file[dataset_type]['class'])
            conf_train = yml_file[dataset_type]['conf']
            conf_train['path'] = os.path.join(DATA_PATH_PREFIX, conf_train['path'])
            
            if dataset_type.startswith('era5') or dataset_type.startswith('station') or dataset_type.startswith('modis') or dataset_type.startswith('cosmo'):
                if yml_file[dataset_type].get('train_time_start', None) is not None and yml_file[dataset_type].get('train_time_end', None) is not None:
                    ind_train_start = yml_file[dataset_type]['train_time_start']
                    ind_train_end = yml_file[dataset_type]['train_time_end']
                    start_time_train = datetime.strptime(ind_train_start, _TIME_FMT)
                    end_time_train = datetime.strptime(ind_train_end, _TIME_FMT)
                    inds_train_ = [np.datetime64(start_time_train + timedelta(hours=i)) for i in range(int((end_time_train - start_time_train).total_seconds() // 3600) + 1)]
                    if yml_file[dataset_type].get('skip_interval_start', None) is not None and yml_file[dataset_type].get('skip_interval_end', None) is not None:
                        ind_skip_start = yml_file[dataset_type]['skip_interval_start']
                        ind_skip_end = yml_file[dataset_type]['skip_interval_end']
                        start_time_skip = datetime.strptime(ind_skip_start, _TIME_FMT)
                        end_time_skip = datetime.strptime(ind_skip_end, _TIME_FMT)
                        ## drop indices from inds_train_ that fall between skip intervals
                        inds_train_ = [ind for ind in inds_train_ if ind < start_time_skip or ind > end_time_skip]
                    conf_train['inds'] = inds_train_
                else:
                    conf_train['inds'] = inds_train
                conf_val = conf_train.copy()
                if yml_file[dataset_type].get('val_time_start', None) is not None and yml_file[dataset_type].get('val_time_end', None) is not None:
                    ind_val_start = yml_file[dataset_type]['val_time_start']
                    ind_val_end = yml_file[dataset_type]['val_time_end']
                    start_time_val = datetime.strptime(ind_val_start, _TIME_FMT)
                    end_time_val = datetime.strptime(ind_val_end, _TIME_FMT)
                    inds_val_ = [np.datetime64(start_time_val + timedelta(hours=i)) for i in range(int((end_time_val - start_time_val).total_seconds() // 3600) + 1)]
                    conf_val['inds'] = inds_val_
                else:
                    conf_val['inds'] = inds_val
            else:
                conf_train['start_idx'] = yml_file[dataset_type]['start_train']
                conf_train['end_idx'] = yml_file[dataset_type]['end_train']
                if 'wb2_path' in conf_train:
                    conf_train['wb2_path'] = os.path.join(DATA_PATH_PREFIX, conf_train['wb2_path'])
                conf_val = conf_train.copy()
                conf_val['start_idx'] = yml_file[dataset_type]['start_val']
                conf_val['end_idx'] = yml_file[dataset_type]['end_val']

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

            batch_sizes.append(yml_file[dataset_type]['batch_size'])

            dataset_train_obj = data_cls(**conf_train)
            if mask_data_cls:
                # If masking is specified, wrap the dataset with MaskDataset
                dataset_train_obj = mask_data_cls(dataset_obj=dataset_train_obj, return_untouched_sample=True, **mask_data_params)
                dataset_train.append(dataset_train_obj)
            else:
                dataset_train.append(dataset_train_obj)
            dataset_val.append(data_cls(**conf_val))
            
    ## keep only unique var names and remove repetitions in tuple. (Note: set() op. is not deterministic.)
    surf_vars = tuple(sorted(set(surf_vars)))
    atmos_vars = tuple(sorted(set(atmos_vars)))
    static_vars = tuple(sorted(set(static_vars)))
    if args.devices > 1:
        dist.init_process_group(backend=args.backend)
    if len(dataset_train) == 1:
        dataloader_train = StatefulDataLoader(
            dataset_train[0], 
            sampler=StatefulDistributedSampler(dataset_train[0], seed=args.seed, drop_last=True) if args.devices > 1 else None,
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,  
            persistent_workers=True,
            collate_fn=collate_fn_train,
            prefetch_factor=4,
        )
        dataloader_val = DataLoader(
            dataset_val[0],
            sampler=DistributedSampler(dataset_val[0], shuffle=False, drop_last=True) if args.devices > 1 else None,
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            # shuffle=False,  # Ensure deterministic for validation
            drop_last=True,
            pin_memory=True,  
            persistent_workers=True,
            collate_fn=collate_fn_val  
        )
        if args.dump_datasampler_indices and args.devices > 1:
            logging_utils.save_sampled_indices_across_ranks(dataloader_train.sampler, seed=args.seed, rank=int(dist.get_rank()), output_dir=os.path.join(args.log_dir, 'data_sampler_indices'))  # Save sampled indices for the first epoch
            logging.info(f"saved sampler indices for rank: {dist.get_rank()}.")
    else:
        dataloader_train = dataset.StatefulMultiDatasetLoader(
            datasets=dataset_train,
            samplers=[StatefulDistributedSampler(ds, seed=args.seed, drop_last=True) for ds in dataset_train] if args.devices > 1 else None,
            batch_sizes=batch_sizes,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fns=[collate_fn_train for _ in dataset_train],
        ) 
        
        dataloader_val = dataset.StatefulMultiDatasetLoader(
            datasets=dataset_val,
            samplers=[StatefulDistributedSampler(ds, shuffle=False, drop_last=True) for ds in dataset_val] if args.devices > 1 else None,
            batch_sizes=batch_sizes,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fns=[collate_fn_val for _ in dataset_val],
        ) 

    return dataloader_train, dataloader_val, surf_vars, atmos_vars, static_vars



def main():
    L.seed_everything(args.seed, workers=True)
    logging_utils.copy_exp_params(log_dir=args.log_dir, config_file=args.config, args=args)
    logging.info("args = %s", args)

    if not args.no_gpu and not torch.cuda.is_available():
        logging.info("GPU training is requested, but no GPU device available.")
        sys.exit(1)


    dataloader_train, dataloader_val, surf_vars, atmos_vars, static_vars = load_data(
        args.dataset_config_path, args.data_sources, args.mask_config_path,
    )
    
    ## Need to check if last.ckpt exists when resume args is passed. If not, one should disable "Resume" status and still consider loading from aurora weights if the other args permit.
    if args.resume: 
        ckpt_fname = os.path.join(args.log_dir, "last.ckpt")
        if os.path.isfile(ckpt_fname):
            trainer_fit_ckpt_path = ckpt_fname
            if SUPPORTS_WEIGHTS_ONLY:
                checkpoint = torch.load(trainer_fit_ckpt_path, weights_only=False)
            else:
                checkpoint = torch.load(trainer_fit_ckpt_path)
            if 'dataloader_state' in checkpoint.keys():
                dataloader_train.load_state_dict(checkpoint['dataloader_state'])
            else:
                logging.info("Warning: No dataloader state found in checkpoint. Training will start observations from scratch!")
        else:
            logging.info(
                f"\n\n\n--resume is passed, but could not find ckpt at {ckpt_fname}. Starting from scratch.\n\n\n"
            )
            trainer_fit_ckpt_path = None
            args.resume = False
    else:
        trainer_fit_ckpt_path = None
        
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

    #teacher
    aurora = ESFMEncoder(
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
        drop_path=0.2,
        num_ensemble = 1,  # Number of ensemble members
        variable_aggregation= False,
        use_resolution_specific_patch_tokenizers = False,
        disable_flashattention=True,
        stabilise_level_agg=False,
        add_qk_norm_to_swin3d=False,
    )
    
    # student
    model = ESFMEncoder(
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
        token_aggregation_attn_drop_rate=args.token_aggregation_attn_drop_rate,
        drop_path=0.2,
        num_ensemble = args.num_ensemble,  # Number of ensemble members
        variable_aggregation= args.variable_aggregation,
        use_resolution_specific_patch_tokenizers = args.use_resolution_specific_patch_tokenizers,
        disable_flashattention=args.disable_flashattention,
        stabilise_level_agg=args.stabilise_level_agg,
        add_qk_norm_to_swin3d=args.add_qk_norm_to_swin3d,
    )
    
    path_esfm_encoder_pretrained_weights = None # path to local copy of pretrained esfm encoder weights. If None, will load from HuggingFace Hub.
    if path_esfm_encoder_pretrained_weights is not None and os.path.isfile(path_esfm_encoder_pretrained_weights):
        path = path_esfm_encoder_pretrained_weights
        aurora.load_state_dict(torch.load(path, map_location=next(aurora.parameters()).device), strict=True)  # Load local pretrained weights
        logging.info(f"Pretrained weights are loaded from local path: {path}")
    else:
        path = hf_hub_download(repo_id="microsoft/aurora", filename=hf_pretrain_fname) # float32
        aurora.load_checkpoint_local(path, strict=False) 
        logging.info(f"Pretrained weights are loaded from HuggingFace Hub: {path}")
    
    # Initially load the pretrained aurora by default.
    if args.load_aurora_pretrain_weights is not None and not args.resume:
        if path == path_esfm_encoder_pretrained_weights:
            model.load_state_dict(torch.load(path, map_location=next(model.parameters()).device), strict=False)  # Load local pretrained weights
        else:
            model.load_checkpoint_local(path, strict=False)  # Load pretrained weights from HuggingFace Hub
        
    logging.info("Pretrained weights are loaded from %s", path)

    ## setup loss function
    loss_obj = losses.KD_Loss_activations(criterion='l1',)
    
    ## setup lightning module & trainer
    class LightningModule(L.LightningModule):
        def __init__(self, net, teacher, loss_fn, **kwargs):
            super().__init__()
            self.net = net
            self.teacher = teacher
            # Ensure teacher model is not part of the trainable parameters
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()  # Set teacher model to evaluation mode
            self.loss_fn = loss_fn
            self.lr_scheduler_interval = kwargs.pop('lr_scheduler_interval', 'step')
            self.batch_size = kwargs.pop('batch_size', None)
            self.learning_rate = kwargs.pop('learning_rate', 5e-4)  # Changed base learning rate to 5e-4
            self.warmup_steps = kwargs.pop('warmup_steps', 1000)    # 1k warmup steps
            self.weight_decay = kwargs.pop('weight_decay', 5e-6)    # AdamW weight decay
            for key, val in kwargs.items():
                setattr(self, key, val)
            self.save_hyperparameters(ignore=['net', 'teacher', 'loss_fn',])
            self.worst_metrics_train, self.worst_metrics_val, self.worst_metrics_test = {}, {}, {}
            self.is_ybatch_images_logged = False
            
        def on_save_checkpoint(self, checkpoint):
            """Remove teacher params from checkpoint state_dict"""
            if 'state_dict' in checkpoint:
                keys_to_pop = [k for k in checkpoint['state_dict'] if k.startswith('teacher.')]
                for k in keys_to_pop:
                    checkpoint['state_dict'].pop(k, None)
            super().on_save_checkpoint(checkpoint)
            
        def load_state_dict(self, checkpoint_state_dict, strict: bool = True):
            """
            Custom logic to load state_dict while ignoring self.teacher
            Student model weights (self.net) are loaded from checkpoint_state_dict as usual.
            """
            # Get the state_dict of the current self.teacher (these are the pre-loaded weights)
            # Add the "teacher." prefix as Lightning expects for submodules.
            pre_loaded_teacher_state_with_prefix = {"teacher." + k: v for k, v in self.teacher.state_dict().items()}

            # Initialize the final state_dict that will be passed to super().load_state_dict()
            final_state_to_load = {}

            # 1. Copy non-teacher keys from the checkpoint_state_dict
            num_student_keys_from_ckpt = 0
            teacher_keys_in_ckpt_count = 0
            for k, v in checkpoint_state_dict.items():
                if not k.startswith("teacher."):
                    final_state_to_load[k] = v
                    num_student_keys_from_ckpt +=1
                else:
                    teacher_keys_in_ckpt_count +=1
            
            if self.global_rank == 0:
                logging.info(f"load_state_dict: Processing {num_student_keys_from_ckpt} non-teacher keys from checkpoint.")
                if teacher_keys_in_ckpt_count > 0:
                    logging.info(f"load_state_dict: Found {teacher_keys_in_ckpt_count} teacher keys in checkpoint; these will be ignored in favor of pre-loaded teacher weights.")


            # 2. Add/overwrite with the pre-loaded teacher's state.
            # This ensures that 'final_state_to_load' has all necessary 'teacher.' keys,
            # but their values are from the already configured self.teacher.
            final_state_to_load.update(pre_loaded_teacher_state_with_prefix)
            if self.global_rank == 0:
                logging.info(f"load_state_dict: Ensured all {len(pre_loaded_teacher_state_with_prefix)} pre-loaded teacher keys are set for loading.")
            
            return super().load_state_dict(final_state_to_load, strict=strict)

        def on_train_start(self):
            """Initialize timing variables when training starts"""
            self.last_step_time = time.time()

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
            self.net.parameters(),  # Use only the parameters of self.net
            lr=self.learning_rate,  # base_lr = 5e-4
            weight_decay=self.weight_decay  # weight decay = 5e-6
            )

            # Calculate total training steps
            total_steps = self.trainer.estimated_stepping_batches 

            # 1. Linear warmup, from 1e-8 to base_lr, within warmup_steps
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-8/self.learning_rate, end_factor=1.0, total_iters=self.warmup_steps)

            # 2. Cosine decay, decrease from base_lr to 0.1x base_lr
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - self.warmup_steps, eta_min=self.learning_rate * 0.1)

            # Combine scheduler
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_steps])

            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
            }


        def forward(self, x):
            return self.net.forward(x)
        
        def _get_system_metrics(self):
            """Gather system metrics including GPU, CPU, memory, and throughput."""
            try:
                # Calculate throughput
                current_time = time.time()
                batch_size = self.trainer.train_dataloader.batch_size
                world_size = self.trainer.world_size
                step_elapsed_time = current_time - self.last_step_time
                step_throughput = (batch_size * world_size) / step_elapsed_time
                        
                # Prepare metrics dictionary with proper WandB formatting
                metrics = {
                    'performance/throughput': step_throughput,  # Changed naming for better WandB organization
                    'performance/cpu_usage': psutil.cpu_percent(),
                    'performance/memory_usage_percent': psutil.virtual_memory().percent,
                    'performance/memory_used_gb': psutil.virtual_memory().used / (1024**3),
                    'performance/memory_available_gb': psutil.virtual_memory().available / (1024**3)
                }
                
                # Add GPU metrics
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        metrics.update({
                            f'performance/gpu_{i}/memory_used_mb': torch.cuda.memory_allocated(i) / 1024**2,
                            f'performance/gpu_{i}/memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024**2,
                            f'performance/gpu_{i}/max_memory_mb': torch.cuda.max_memory_allocated(i) / 1024**2
                        })
                
                self.last_step_time = current_time
                return metrics
                
            except Exception as e:
                print(f"Error collecting system metrics: {str(e)}")
                return {}
            
        def _check_for_nan_weights(self):
            is_nans = False
            for name, param in self.named_parameters():
                if torch.isnan(param).any():
                    logging.error(f"NaN detected in layer: {name}")
                    is_nans = True
                if torch.isinf(param).any():
                    logging.error(f"Inf detected in layer: {name}")
                    is_nans = True
            return is_nans

        def training_step(self, batch, batch_idx):
            if args.kill_on_nan_detection and batch_idx % args.check_interval_nan_model_weights == 0:
                if self._check_for_nan_weights():
                    logging.error("NaN or Inf detected in model weights. Stopping training.")
                    raise ValueError("NaN or Inf detected in model weights. Stopping training.")
            
            # batch_obj_x = self._make_batch(batch)
            if isinstance(batch, list):
                batch_obj_x = batch[0]
                batch_obj_untouched = batch[1]
            else:
                batch_obj_x = batch
                batch_obj_untouched = batch

            # Forward pass
            with torch.no_grad():
                teacher_logits = self.teacher.forward(batch_obj_untouched)
            student_logits = self.net.forward(batch_obj_x)
            del batch_obj_x

            # Main task loss
            task_loss = self.loss_fn(pred_batch=student_logits, target_batch=teacher_logits)
            total_loss = task_loss
            loss_dict = {'KD_loss': task_loss}
            del teacher_logits 
            del student_logits

            # Get system metrics and add to loss_dict 
            system_metrics = self._get_system_metrics()
            loss_dict.update(system_metrics)

            train_keys_replicated = {}
            for key in loss_dict: 
                train_keys_replicated[f'train/{key}'] = loss_dict[key]
            train_keys_replicated['loss_train'] = total_loss
            loss_dict.update(train_keys_replicated)

            # Log all losses
            self.log_dict(loss_dict, batch_size=self.batch_size, sync_dist=True, prog_bar=True)

            return total_loss

        def validation_step(self, val_batch, batch_idx):
            batch_obj_x = val_batch
            teacher_logits = self.teacher.forward(batch_obj_x)
            student_logits = self.net.forward(batch_obj_x)
            del batch_obj_x
            
            # Main task loss
            task_loss = self.loss_fn(pred_batch=student_logits, target_batch=teacher_logits)
            
            total_loss = task_loss
            loss_dict = {'KD_loss': task_loss}
            
            # Log metrics
            loss_dict = {f'{key}_val': value for key, value in loss_dict.items()}
            loss_dict['loss_val'] = total_loss
            self.log_dict(loss_dict, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        
        def on_after_backward(self):
            """Override to check for NaN in gradients if enabled."""
            if args.kill_on_nan_detection:
                # Check for NaN in loss and skip optimizer step if detected.
                nan_or_inf_found = False
                for param in self.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            logging.error("NaN detected in gradients! Resetting gradients for step...")
                            nan_or_inf_found = True
                            break
                        if torch.isinf(param.grad).any():
                            logging.error("Inf detected in gradients! Resetting gradients for step...")
                            nan_or_inf_found = True
                            break
                if nan_or_inf_found:
                    raise ValueError("NaN or Inf detected in gradients. Stopping training.")
                
            super().on_after_backward()  # Call the parent method to ensure any additional behavior is executed
        
        def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_closure=None,
            optimizer_idx=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
        ):  
            # Run the closure to get the loss and compute gradients
            if optimizer_closure is not None:
                optimizer_closure()
            
            # Clip gradients using FSDP's clip_grad_norm_: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
            if hasattr(self, 'net'):
                # Check if we're using FSDP strategy
                using_fsdp = isinstance(self.trainer.strategy, FSDPStrategy)
                if using_fsdp:
                    # Get the FSDP wrapper from the strategy
                    fsdp_wrapper = self.trainer.strategy.model
                    if args.log_norms:
                        pre_clip_norm = fsdp_wrapper.clip_grad_norm_(max_norm=float('inf')) 
                    fsdp_wrapper.clip_grad_norm_(max_norm=args.max_grad_norm)  # Apply clipping
                    
                    if args.log_norms:
                        if int(self.trainer.global_step) % args.log_norm_every_n_steps == 0:
                            
                            # Measure the norm again to confirm it's now ≤ 1.0
                            with torch.no_grad():
                                grad_norms = [
                                    torch.norm(p.grad.detach(), 2) 
                                    for p in fsdp_wrapper.parameters() 
                                    if p.grad is not None
                                ]
                                if len(grad_norms) > 0:
                                    post_clip_norm = torch.norm(torch.stack(grad_norms), 2)
                                else:
                                    post_clip_norm = torch.tensor(0.0, device=next(fsdp_wrapper.parameters()).device)

                            if is_rank0:
                                grad_metrics = log_gradient_norms(fsdp_wrapper)
                                weight_metrics = log_weight_norms(fsdp_wrapper)
                                self.log('grad_overall/grad_norm_pre_clip', pre_clip_norm)
                                self.log('grad_overall/grad_norm_post_clip', post_clip_norm)
                                self.log_dict(grad_metrics, sync_dist=False)
                                self.log_dict(weight_metrics, sync_dist=False)
                else:
                    net = self.net
                    # Fallback to regular gradient clipping
                    parameters = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
                    if parameters:
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.max_grad_norm)
                        
                    if args.log_norms:
                        if int(self.trainer.global_step) % args.log_norm_every_n_steps == 0:
                            
                            # Measure the norm again to confirm it's now ≤ 1.0
                            with torch.no_grad():
                                post_clip_norm = torch.norm(torch.stack([
                                    torch.norm(p.grad.detach(), 2) 
                                    for p in net.parameters() 
                                    if p.grad is not None
                                ]), 2)

                            if is_rank0:
                                grad_metrics = log_gradient_norms(net)
                                weight_metrics = log_weight_norms(net)
                                self.log('grad_overall/grad_norm_post_clip', post_clip_norm)
                                self.log_dict(grad_metrics, sync_dist=False)
                                self.log_dict(weight_metrics, sync_dist=False)
            
            # Update parameters
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()

    # Initialize LightningModule
    model_lightning = LightningModule(
        net=model, 
        teacher=aurora,
        loss_fn=loss_obj.get_loss,  
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=5e-6,
        warmup_steps=args.lr_warmup_steps,
    )

    class StatefulDataLoaderCallback(Callback):
        """
        PyTorch Lightning callback to save and restore dataloader state during training.

        Args:
            dataloader (DataLoader): The dataloader instance to manage
            checkpoint_dir (str): Directory to save dataloader state checkpoints
        """
        def __init__(self, dataloader: DataLoader, checkpoint_dir: str = './dataloader_checkpoints'):
            super().__init__()
            self.dataloader = dataloader
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)

        def _save_dataloader_state(self, trainer, checkpoint_path):
            """Save dataloader state to a file"""
            state_path = os.path.join(self.checkpoint_dir, f"dataloader_state_{trainer.global_step}.pkl")
            with open(state_path, 'wb') as f:
                pickle.dump(self.dataloader.state_dict(), f)
            return state_path

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            """Save dataloader state when a checkpoint is saved"""
            # Only save on rank 0 to prevent file conflicts
            if trainer.is_global_zero:
                # Check if the checkpoint is triggered by modelcheckpoint_callback_regular_step_save
                for callback in trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint) and callback == modelcheckpoint_callback_regular_step_save:
                        # Add the dataloader state in the checkpoint
                        checkpoint['dataloader_state'] = self.dataloader.state_dict()
                        break

    ## Define Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)
    modelcheckpoint_callback_regular_step_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-step-{step}-{loss_train:.2f}", 
        every_n_train_steps=100, 
        save_last=True,
        save_top_k = -1, ## save all ckpts.
    )
    modelcheckpoint_callback_regular_epoch_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-{epoch}-{loss_train:.2f}", 
        save_on_train_epoch_end=True, 
        save_last=True
    )
    modelcheckpoint_callback_best_val_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_best_val_ckpt-{epoch}-{loss_val:.2f}", 
        monitor="loss_val", 
        save_top_k=3, 
        mode='min', 
        save_on_train_epoch_end=False
    )
    # 1) define a small helper to pull batch‐size out of your Batch object:
    def batch_size_fn(batch):
        if isinstance(batch, list):
            batch_obj_x = batch[0]
        else:
            batch_obj_x = batch
        surf_var = batch_obj_x.surf_vars.get('2t', next(iter(batch_obj_x.surf_vars.values())))
        return surf_var.shape[0]

     # Configure progress bar for SLURM logs (no ANSI escape codes)
    if not sys.stdout.isatty():
        # Non-interactive mode: print progress line-by-line without ANSI codes
        progress_bar = TQDMProgressBar(refresh_rate=10, process_position=0)
    else:
        # Interactive mode: use default progress bar
        progress_bar = TQDMProgressBar()

    callbacks = [
        progress_bar,
        modelcheckpoint_callback_regular_step_save,
        modelcheckpoint_callback_regular_epoch_save, 
        modelcheckpoint_callback_best_val_save,
        StatefulDataLoaderCallback(dataloader=dataloader_train, checkpoint_dir=args.log_dir),
        lr_monitor,
    ]

    ## Setup Loggers
    logger = WandbLogger(
        save_dir=args.log_dir,
        entity=args.wnb_entity,
        name=args.wnb_name,
        project=args.wnb_project,
        id=args.wnb_id,
        log_model=False,
        save_code=True,
        resume='allow',
        mode=args.wnb_mode,
        config=args
    )

    ## Setup and Launch Lightning Trainer
    deterministic_trainer = False  # Might make training slower
    check_val_every_n_epoch = 1  # Use val_check_interval if you want to run val every N steps
    total_train_minibatches = int(len(dataloader_train))
    print(f'get_total_gpus returns {get_total_gpus()}. len(dataloader_train): {len(dataloader_train)}')
    if total_train_minibatches >= 900:
        val_check_interval = 900
    else:
        val_check_interval = total_train_minibatches
    if is_rank0:
        print(f'val_check_interval: {val_check_interval}.')
    log_every_n_steps = args.log_every_n_steps  # How often to add logging rows
    max_epochs = args.epochs
    accelerator = 'gpu' if not args.no_gpu else 'cpu'
    devices = args.devices  # Number of GPUs on each node
    num_nodes = args.num_nodes  # Number of nodes. Total GPUs = num_nodes x devices

    ### Strategy
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not hasattr(args, 'strategy'):
        strategy_str = 'auto'
    else:
        strategy_str = args.strategy.lower()
    if num_gpus > 1:
        if strategy_str == 'full_fsdp':
        # # FSDPStrategy with SHARD GRAD OP
            fsdp_strategy = FSDPStrategy(
                cpu_offload=True, # Keeps optimizer states and gradients offloaded to CPU
                sharding_strategy="FULL_SHARD", # Shards gradients and optimizer states, parameters are replicated
                backward_prefetch="SHARD", # Changed from None to "SHARD" if applicable, or "BACKWARD_PRE"
                use_orig_params=True,
                timeout = timedelta(seconds=6000), # set NCCL timeout to 100 mins
                process_group_backend=args.backend
            )
            strategy = fsdp_strategy
        else:
            strategy = DDPStrategy(
                find_unused_parameters=True,  # Set to True if your model has unused parameters
                process_group_backend=args.backend
            )
    else:
        strategy = 'auto'
    if is_rank0:
        logging.info(f"Strategy: {strategy}, num_gpus: {num_gpus}.")

    # Keep precision as float32 in trainer
    trainer = L.Trainer(
        accelerator=accelerator, 
        devices=devices, 
        num_nodes=num_nodes, 
        strategy=strategy, 
        precision='32-true',  # Keep this as 32-bit
        deterministic=deterministic_trainer, 
        callbacks=callbacks, 
        check_val_every_n_epoch=check_val_every_n_epoch, 
        log_every_n_steps=log_every_n_steps, 
        logger=logger, 
        min_epochs=10, 
        max_epochs=max_epochs, 
        # profiler="pytorch", 
        profiler=None,
        enable_progress_bar=True, 
        num_sanity_val_steps=2, 
        use_distributed_sampler=False,
    )
    logging.info(f"trainer state fn: {trainer.state.fn}, status: {trainer.state.status}.")
        
    trainer.fit(model_lightning, dataloader_train, dataloader_val, ckpt_path=trainer_fit_ckpt_path)

    wandb.finish()
    
    logging.info("Training is completed.")

if __name__ == "__main__":
    main()
