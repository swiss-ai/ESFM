# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import argparse
import yaml

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
        yaml_args = argparse.Namespace()
        yaml_args.__dict__.update(config)
    return yaml_args


def get_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False) # forbid abbreviations for safer behaviour with parse_known_args
    parser.add_argument("--config", help="Load settings from yaml.")
    parser.add_argument(
        "--dataset_config_path", 
        default='dataset_config.yaml',
        help="Load dataset configs from yaml.")
    parser.add_argument("--no_gpu", action="store_true", default=False, help="Explicitly use CPU [default: uses gpu]")
    parser.add_argument("--num_nodes", type=int, default=1, help="num nodes to train on")
    parser.add_argument("--devices", type=int, default=1, help="num GPU devices on each node to train on")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_workers", type=int, default=1, help="#threads to run for dataloaders")
    parser.add_argument("--backend", type=str, default='nccl', help="Backend for distributed trianing ")
    parser.add_argument(
        "--mask_config_path",
        type=str,
        default='masking_config.yaml',
        help="Path to the mask config yaml file. If None, no masking is applied. [default: None]",
    )
    parser.add_argument(
        "--mask_config_type",
        type=str,
        default=None,
        help="Masking config type to use. [default: None]",
    )
    parser.add_argument("--log_dir", default="checkpoints/", help="Log dir [default: log]")
    parser.add_argument("--data", type=str, default="./data", help="dataset path")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch to run [default: 200]")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size during training. Warning: dataset classes override it. [default: 1]"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Initial learning rate [default: 5e-4]"
    )
    parser.add_argument("--optimizer", default="adamW", help="adam or sgd [default: adamW]")
    parser.add_argument(
        "--opt_eps", type=float, default=1e-6, help="AdamW epsilon [default: 1e-6]"
    )
    parser.add_argument(
        "--opt_betas",
        type=float,
        nargs=2,                 # expect two floats
        default=(0.9, 0.95),     # default tuple
        metavar=("BETA1", "BETA2"),
        help="AdamW betas (beta1, beta2)."
    )
    parser.add_argument(
        '--reset_optimizer',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to reset optimizer when resuming from a checkpoint. [default: False]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=1000,
        help="Number of steps for learning rate warmup. [default: 1000]",  
    )
    parser.add_argument(
        "--constant_learning_rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use constant learning rate instead of cosine annealing. [default: False]",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume training using {log_dir}/last.ckpt.",
    )
    parser.add_argument(
        "--strict_loading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict=False when changing network configuration mid-training (resuming lightning.trainer). [default: True]",
    )
    parser.add_argument(
        "--load_dataset_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the DataLoader status from checkpoint. If False, Dataloader and sampler per rank will be re-initialized. [default: True]",
    )
    parser.add_argument(
        "--dump_datasampler_indices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dump the sampled indices of the dataset sampler to a csv file. [default: False]",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch. [default: 2]"
    )
    # load checkpoint for inference
    parser.add_argument('--ckpt_name', type=str, 
                    default="last.ckpt",
                    help='Name of the checkpoint file to load')
    # dataset str_to use for validation
    parser.add_argument(
        "--val_data_sources",
        nargs='+',
        default=["era5"],
        help="Which dataset split & vars to use for validation. [default: [era5]]",
    )
    parser.add_argument(
        "--test_data_sources",
        nargs='+',
        default=["era5_22to25"],
        help="Which dataset split & vars to use for testing. [default: [era5_22to25]]",
    )
    parser.add_argument(
        "--vars_mask_during_testing",
        type=str,
        default=None,
        help='JSON string of variables to mask during testing. Format: [["2t", "surf_var"], ["10u", "surf_var"], ["t", "atmos_var"]]. If None, no masking is applied. [default: None]',
    )
    parser.add_argument(
        "--plevs_mask_during_testing",
        nargs='+',
        type=int,
        default=None,
        help="List of pressure levels to mask during testing. If None, no masking is applied. [default: None]",
    )
    parser.add_argument(
        "--mask_regions_during_testing",
        nargs='+',
        type=str,
        default=None,
        help="sequence of regions to mask during testing. Options: [None, 'switzerland', 'europe', 'usa']. If None, no masking is applied. [default: None]",
    )
    parser.add_argument(
        "--mask_multiple_during_testing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Will mask input according to all of the above args simultaneously. [default: False]",
    )
    parser.add_argument(
        "--load_aurora_pretrain_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize model weights to ESFM pretrained weights, where applicable. [default: True]",
    )
    parser.add_argument(
        "--load_custom_pretrain_weights_str",
        type=str,
        default=None,
        help="Load custom pretrained weights from an absolute path. If None, no custom weights are loaded. [default: None]",
    )
    parser.add_argument(
        "--stabilise_level_agg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Applies additional layer norm to perceiver modules. [default: False]",
    )
    parser.add_argument(
        "--act_checkpointing_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for encoder. [default: True]",
    )
    parser.add_argument(
        "--act_checkpointing_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for backbone. [default: True]",
    )
    parser.add_argument(
        "--act_checkpointing_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for decoder. [default: True]",
    )
    parser.add_argument(
        "--var_attn_chunk_size",
        type=int,
        default=0,
        help="Chunk size (along the flattened batch*grid or batch*levels*grid dimension) for "
             "the encoder's checkpointed variable-attention modules. Larger values mean fewer, "
             "bigger sub-calls per checkpoint (faster, more GPU memory) at a lower value "
             "meaning smaller peak activation memory (safer, slower). Pass 0 to disable "
             "chunking entirely. [default: 4096]",
    )
    parser.add_argument(
        "--level_decoder_chunk_size",
        type=int,
        default=0,
        help="Same as --var_attn_chunk_size but for the decoder's level_decoder call. "
             "[default: 4096]",
    )
    parser.add_argument("--wnb_entity", type=str, default="esfm", help="W&B entity name")
    parser.add_argument("--wnb_project", type=str, default="esfm_era5", help="W&B project name")
    parser.add_argument("--wnb_name", type=str, default="", help="W&B run name")
    parser.add_argument("--wnb_id", type=str, default=None, help="W&B run id")
    parser.add_argument(
        "--wnb_mode", type=str, default="online", help="W&B mode. use online or disabled"
    )
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="log freq for wandb")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--log_norms", action=argparse.BooleanOptionalAction, default=False, help="Logs gradient and weight norms of model [default: False]",)
    parser.add_argument("--log_norm_every_n_steps", type=int, default=100, help="log freq for weight and gradient norms on wandb")
    parser.add_argument("--log_val_predictions_as_images", action=argparse.BooleanOptionalAction, default=False, help="Logs the first sample in validation step as pictures to W&B [default: False]",)
    ## ensemble args
    parser.add_argument("--num_ensemble", type=int, default=1, help="number of ensembles")
    parser.add_argument(
        "--mae_on_ensemble_mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute MAE on ensemble mean instead of per-member. [default: True]",
    )
    parser.add_argument("--mae_weight", type=float, default=1.0, help="mae_weight")
    parser.add_argument("--nll_weight", type=float, default=0.0, help="nll_weight")
    parser.add_argument("--crps_weight", type=float, default=0.0, help="crps_weight")
    parser.add_argument("--kernel_crps_weight", type=float, default=0.0, help="kernel_crps_weight")
    parser.add_argument("--almost_fair_crps_alpha", type=float, default=1.0, help="alpha parameter [0, 1] for almost fair crps loss. If 1.0, it's fair crps. If 0.95, it's afCRPS from AIFS-CRPS. [default: 1.0]")
    parser.add_argument("--stats_loss_weight", type=float, default=0.0, help="nll_wstats_loss_weighteight")
    parser.add_argument("--latitude_weight", action=argparse.BooleanOptionalAction, default=True, help="Whether to use latitude_weight [default: True]",)
    parser.add_argument(
        "--use_legacy_tails",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Instead of the new AdaLN based ensembles, use legacy detokenizer tails as ensembles. [default: False]",
    )
    parser.add_argument("--ensemble_adaln_scale_bias", type=float, default=1.0, help="Scale bias for AdaLN layers.")
    parser.add_argument(
        "--loss_config_path", 
        default='loss_config.yaml',
        help="Load loss weights for each surface variable from yaml.")
    parser.add_argument(
        "--loss_config_name",
        type=str,
        default='default',
        help="Which loss config to use from the loss_config_path yaml."
    )
    parser.add_argument("--check_interval_nan_model_weights", type=int, default=100, help="Frequency to check for NaN or Inf in model weights. [default: 100]")
    parser.add_argument(
        "--kill_on_nan_detection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Kill the process if NaN or Inf is detected in gradients or model weights. [default: False]",
    )
    parser.add_argument("--strategy", type=str, default='full_fsdp', help="training strategy.")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable find_unused_parameters in DDP if that's the training strategy. [default: False]",
    )
    parser.add_argument(
        "--str_architecture_size",
        type=str,
        default="large",
        choices=["small", "large",],
        help="Size of the architecture. Options: [small ,large] [default: large]", ## consider expanding with tiny, small, base, large, huge for future, in case we want to use them.
    )
    parser.add_argument(
        "--use_resolution_specific_patch_tokenizers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable multi-(de-)tokenizers [default: False]",
    )
    parser.add_argument(
        "--variable_aggregation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable variable aggregation [default: False]",
    )
    parser.add_argument(
        "--axial_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable axial attention. This works if variable_aggregation is True [default: True]",
    )
    parser.add_argument(
        "--disable_flashattention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disables all flash attention implementations and uses native pytorch self-attention instead. [default: False]",
    )
    parser.add_argument(
        "--add_qk_norm_to_swin3d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Applies additional qk layer norm to swin3d. [default: False]",
    )    

    parser.add_argument(
        "--data_sources",
        nargs='+',
        default=['era5'],
        help="List of data sources to use.",
    )
    parser.add_argument(
        "--data_source_ratios",
        nargs='+',
        type=int,
        default=None,
        help="List of ratios for each data source. Must match the length of data_sources. Example: [10, 1] means for each sample from the second data source, 10 samples from the first data source will be used. [default: None - equal ratios]",
    )
    parser.add_argument(
        "--grid_deg_delta",
        type=float,
        default=0.01,
        help=f"Grid degree delta for position encoding. Use 0.00001 if you plan to finetune model with station data in future. [Default: 0.01]",
    )
    parser.add_argument(
        "--absolute_time_embedding_in_minutes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use minute resolution for absolute time embedding instead of hourly resolution. [default: False]",
    )
    parser.add_argument(
        "--add_token_pos_embedding_in_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add token positional embedding in the decoder. [default: False]",
    )
    parser.add_argument(
        "--num_max_ensembles",
        type=int,
        default=1000,
        help="Maximum number of ensembles the model can predict. [default: 1000]",
    )
    parser.add_argument(
        "--f",
        type=str,
        default='',
        help="does nothing, for notebook compatibility",
    )
    return parser
    
def parse_known_args():
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    if args.config:
        print(f"Loading config from {args.config}. Will overwrite any command line arguments with yaml content.")
        yaml_args = parse_config(args.config)
        args.__dict__.update({**args.__dict__, **yaml_args.__dict__})

    return args, unknown

def parse_args(custom_args=None):
    parser = get_parser()
    if custom_args is not None:
        args, _ = parser.parse_known_args(custom_args)
    else:
        args, _ = parser.parse_known_args()

    if hasattr(args, 'config') and args.config:
        print(f"Loading config from {args.config}. Will overwrite any command line arguments with yaml content.")
        yaml_args = parse_config(args.config)
        args.__dict__.update({**args.__dict__, **yaml_args.__dict__})

    return args