# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os, glob
import torch
from esfm.model.esfm_encoder_only import ESFMEncoder
from esfm.model.esfm import ESFM
from config import parse_args
from huggingface_hub import hf_hub_download
import numpy as np
import yaml
from natsort import natsorted

DATA_PATH_PREFIX = os.getenv("ESFM_DATA_PATH_PREFIX", "/capstor/store/cscs/").rstrip("/") + "/"
def load_data(yaml_path, datasets_type):
    """Load datasets based on YAML configuration."""
    surf_vars, atmos_vars, static_vars = (), (), ()
    for dataset_type in datasets_type:
        with open(yaml_path, 'r') as file:
            yml_file = yaml.safe_load(file)
            conf_train = yml_file[dataset_type]['conf']
            conf_train['path'] = os.path.join(DATA_PATH_PREFIX, conf_train['path'])

            surf_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                              for var in yml_file[dataset_type]['surf_vars']] 
                              if 'variable_name_mapping' in yml_file[dataset_type]['conf'] 
                              else yml_file[dataset_type]['surf_vars'])
            atmos_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['atmos_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['atmos_vars'])
            static_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['static_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['static_vars'])

    return surf_vars, atmos_vars, static_vars


def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    surf_vars, atmos_vars, static_vars = load_data(
        args.dataset_config_path, args.data_sources
    )
    if args.mask_config_type is None:
        str_is_masked = ""
    else:
        str_is_masked = f"masked_"
    
    str_architecture_size = args.str_architecture_size
    if str_architecture_size == "small":
        encoder_depths = (2,6,2)
        encoder_num_heads = (4,8,16)
        decoder_depths = (2, 6, 2)
        decoder_num_heads = (16, 8, 4)
        embed_dim = 256
        num_heads = 8
        hf_pretrain_fname = 'aurora-0.25-small-pretrained.ckpt'
        str_ckpt_name_out_prepend = 'small_'
    elif str_architecture_size == "large":
        encoder_depths = (6, 10, 8)
        encoder_num_heads = (8, 16, 32)
        decoder_depths = (8, 10, 6)
        decoder_num_heads = (32, 16, 8)
        embed_dim= 512
        num_heads = 16
        hf_pretrain_fname = 'aurora-0.25-pretrained.ckpt'
        str_ckpt_name_out_prepend = ''
    else:
        raise ValueError(f"Unknown architecture size: {str_architecture_size}. Choose 'small' or 'large'.")

    # Initialize ESFMEncoder (student network) based on config parameters
    student_encoder = ESFMEncoder(
        use_lora=False, 
        autocast=True, # Use AMP (mixed precision to fit to GPU)
        encoder_depths=encoder_depths,
        encoder_num_heads=encoder_num_heads,
        decoder_depths=decoder_depths,
        decoder_num_heads=decoder_num_heads,
        embed_dim=embed_dim,
        num_heads=num_heads,
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        drop_path=0.2,
        num_ensemble = args.num_ensemble,  # Number of ensemble members
        variable_aggregation= args.variable_aggregation,
        axial_attention=args.axial_attention,
        use_resolution_specific_patch_tokenizers = args.use_resolution_specific_patch_tokenizers,
        disable_flashattention=args.disable_flashattention,
        stabilise_level_agg=args.stabilise_level_agg,
        add_qk_norm_to_swin3d=args.add_qk_norm_to_swin3d,
    )
    student_encoder.to(device)
    # Load weights for ESFMEncoder from final checkpoint in log_dir
    ckpt_list = natsorted(glob.glob(os.path.join(args.log_dir, "model_ckpt-step*")))
    ## get ckpt int list:
    d_ckpt_step = {}
    for ckpt in ckpt_list:
        try:
            s = os.path.basename(ckpt).split('=')[1].split("-loss_train")[0]
            step = int(s)
            d_ckpt_step[step] = ckpt
        except:
            print(f'Could not parse step from checkpoint filename: {ckpt}')
            continue
    ## Get the largest ckpt that is a multiple of 10k.
    l_ckpt_step = list(d_ckpt_step.keys())
    l_ckpt_step_10k = [step for step in l_ckpt_step if step % 10000 == 0]
    ckpt_step = l_ckpt_step_10k[-1] 
    encoder_ckpt_path = d_ckpt_step[ckpt_step]
    ## get ckpt step by finding int value in between `model_ckpt-step` and `loss_train` ignoring other strings
    
    logging.info(f"Loading encoder checkpoint from {encoder_ckpt_path} at step {ckpt_step}")

    if os.path.isfile(encoder_ckpt_path):
        lightning_state_dict = torch.load(encoder_ckpt_path, map_location=device)
        student_state_dict = lightning_state_dict['state_dict'] 
        for key in list(student_state_dict.keys()):
            # Remove 'net.' prefix if it exists
            if key.startswith('net.'):
                new_key = key[len('net.'):]
                student_state_dict[new_key] = student_state_dict.pop(key)
        student_encoder.load_state_dict(student_state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Encoder checkpoint not found at {encoder_ckpt_path}")

    # Initialize full ESFM network (teacher network) based on config parameters
    # Default model architecture
        
    full_model = ESFM(
        use_lora=False, 
        autocast=True, # Use AMP (mixed precision to fit to GPU)
        encoder_depths=encoder_depths,
        encoder_num_heads=encoder_num_heads,
        decoder_depths=decoder_depths,
        decoder_num_heads=decoder_num_heads,
        embed_dim=embed_dim,
        num_heads=num_heads,
    )
    full_model.to(device)

    # Load pretrained weights for ESFM
    path = hf_hub_download(repo_id="microsoft/aurora", filename=hf_pretrain_fname) # float32
    full_model.load_checkpoint_local(path, strict=True) ## we do not yet have slt
    
    # Replace the encoder of the full ESFM model with the student ESFM encoder
    full_model.encoder = student_encoder.encoder

    # Save the modified ESFM model weights to log_dir
    modified_model_path = os.path.join(args.log_dir, f"{str_ckpt_name_out_prepend}esfm_with_{str_is_masked}_encoder_step_{ckpt_step}.ckpt")
    torch.save(full_model.state_dict(), modified_model_path)
    logging.info(f"Modified ESFM model with student encoder saved at {modified_model_path}")

if __name__ == "__main__":
    main()
