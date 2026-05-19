# ESFM: Earth System Foundation Model

This repository contains the ESFM training and evaluation code used for atmospheric and Earth-system forecasting experiments.

The codebase is based on the Aurora model stack and extends it with ESFM-specific model components, data pipelines, and experiment scripts (both training and evaluation).

<div align="center">
  <img src="assets/ESFM-schematic-teaser.png" width="800" />
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/3cf8870a-4e5e-450f-af37-bc120cc9b721" width="800" />
</div>


<div align="center">
  <img src="https://github.com/user-attachments/assets/cc309b2c-35a1-41c7-a3d4-c127b12aaa87" width="800" />
</div>


## Contents

- [ESFM: Earth System Foundation Model](#esfm-earth-system-foundation-model)
  - [Contents](#contents)
  - [Project Scope](#project-scope)
  - [Repository Layout](#repository-layout)
  - [Requirements](#requirements)
  - [Model Weights](#model-weights)
  - [Preprint](#preprint)
  - [Configuration System](#configuration-system)
  - [Training](#training)
  - [Inference](#inference)
    - [Single-step inference](#single-step-inference)
    - [Multi-step rollout inference](#multi-step-rollout-inference)
  - [Data Preprocessing](#data-preprocessing)
  - [Station Metrics Evaluation](#station-metrics-evaluation)
  - [Experiment Scripts](#experiment-scripts)
  - [Path Update Helpers](#path-update-helpers)
    - [Update workdir defaults in Slurm scripts](#update-workdir-defaults-in-slurm-scripts)
    - [Update log\_dir base in config files](#update-log_dir-base-in-config-files)
  - [Troubleshooting Notes](#troubleshooting-notes)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Contributing](#contributing)

## Project Scope

This repo provides:

- ESFM model implementations in `esfm/model/`
- Dataset wrappers and mixed-source loading in `utils/dataset.py`
- Training entrypoints for standard ESFM and encoder KD pretraining
- Inference entrypoints for single-step and multi-step rollout forecasting
- Metric computation for station datasets (ECMWF 11k & Weather-5K) using `evaluate_station_metrics.py`; gridded dataset evaluation (ERA5, CMIP6, MODIS) using the [SwissClim_Evaluations (v0.2.0)](https://github.com/swiss-ai/SwissClim_Evaluations/tree/v0.2.0) toolbox.

Important environment assumption:

- Many scripts are configured for CSCS paths (for example `/capstor/store/cscs/...`) and Slurm execution.
- If you run outside CSCS, you must adapt data paths and launcher scripts.

## Repository Layout

- `esfm/`: core package (model, batch representation, rollout helpers, normalization)
- `configs/`: experiment configuration files (model size, optimization, logging, etc.)
- `dataset_config.yaml`: dataset definitions 
- `masking_config.yaml`: optional masking strategies for training
- `loss_config.yaml`: per-variable loss weights
- `train.py`: main ESFM training entrypoint
- `train_encoder_KD.py`: encoder knowledge-distillation training entrypoint
- `inference.py`: single-step inference/evaluation entrypoint
- `inference_rollout.py`: multi-step rollout inference entrypoint
- `evaluate_station_metrics.py`: metrics computation for station experiments from prediction zarr files
- `scripts/training/`: Slurm training scripts to reproduce experiments from the manuscript
- `scripts/inference/`: Slurm inference scripts to reproduce experiments from the manuscript

## Requirements

All experiments have been tested on the Container Engine of CSCS Alps.  
Most of our experiments have ran on an image based off of modulus:24.04, but later verified with physicsnemo:25.03.

## Model Weights

Model weights for the different experiments from the manuscript are available on [Hugging Face (ESFM)](https://huggingface.co/ESFM).

## Preprint

The ESFM preprint is available [[here](https://arxiv.org/abs/2605.00850)].

## Configuration System

CLI arguments are defined in `config.py`.

You usually run with:

- `--config`: experiment yaml in `configs/`
- `--dataset_config_path`: dataset registry yaml (default: `dataset_config.yaml`)
- `--loss_config_path`: variable-weight config (default: `loss_config.yaml`)
- `--mask_config_path` and `--mask_config_type`: optional masking setup

Example config file:

- `configs/config_ESFM_s_enc_KD_nm.yaml`

This sets options such as:

- architecture (`str_architecture_size`)
- devices and workers
- optimization and gradient settings
- logging (`wnb_*`)
- data sources (`data_sources`)

Dataset path root override:

- Entry points (`train.py`, `train_encoder_KD.py`, `inference.py`, `inference_rollout.py`, `merge_esfm_encoder.py`) read `ESFM_DATA_PATH_PREFIX` if set.
- If unset, they fall back to `/capstor/store/cscs/`.
- Keep in mind that you will nevertheless need to adjust the relative paths under `dataset_config.yaml`.

Example:

```bash
export ESFM_DATA_PATH_PREFIX=/your/storage/root
```

Prediction copy destination override:

- `utils/validation_utils.py` reads `ESFM_RESULTS_COPY_BASE` when copying merged inference predictions.
- If unset, it falls back to `/capstor/store/cscs/swissai/a122/ESFM_Results`.

Example:

```bash
export ESFM_RESULTS_COPY_BASE=/your/results/storage/ESFM_Results
```

## Training

For reproducing ESFM variants proposed in our work, please look at the slurm scripts listed under `scripts/training`.  
Further information is provided in detail under [README](scripts/README.md).  
For example, after adapting paths to your custom compute environment, you can run a training as:
```bash
sbatch scripts/training/train_ESFM_s_wm_ri_pre.sh
```

## Inference

### Single-step inference

For a quick interactive ERA5 visualisation, see [notebooks/inference_ESFMs_on_ERA5.ipynb](notebooks/inference_ESFMs_on_ERA5.ipynb). For actual and efficient inference on local datasets, use the Python and Slurm examples below.

```bash
python inference.py --config ./configs/config_ESFM_s_nm.yaml
```

Slurm example:

```bash
sbatch scripts/inference/inference_ESFM_s_nm.sh
```

### Multi-step rollout inference

Rollout uses `inference_rollout.py` and supports custom args such as `--NUM_ROLLOUT_STEPS`.

Example:

```bash
python inference_rollout.py \
  --config ./configs/config_ESFM_s_nm_e11k_lt6h.yaml \
  --NUM_ROLLOUT_STEPS 56
```

Slurm example:

```bash
sbatch scripts/inference/inference_ESFM_s_nm_e11k_eval1_lt6h.sh
```

## Data Preprocessing

CMIP6, MODIS, Weather-5K, and ECMWF 11k datasets have been preprocessed using scripts released under the [SwissClim Data Processing Scripts](https://github.com/swiss-ai/SwissClim_data_processing_scripts) repository.

## Station Metrics Evaluation

After generating predictions (zarr output), compute detailed station metrics with:

```bash
python evaluate_station_metrics.py \
  --p_preds /path/to/predictions.zarr \
  --station_type 11k \
  --p_station /path/to/station_dataset_root \
  --output_dir ./eval_out \
```
## Experiment Scripts

For manuscript-aligned experiment commands and checkpoint notes, see:

- `scripts/README.md`
- `scripts/training/`
- `scripts/inference/`

These scripts are tuned for a specific HPC environment and should be adapted before running elsewhere.

## Path Update Helpers

Two helper scripts are provided to safely bulk-update common path roots.

- `scripts/update_workdir_roots.sh`: updates default `workdir="..."` lines in Slurm scripts under `scripts/training/` and `scripts/inference/`.
- `scripts/update_logdir_roots.sh`: updates `log_dir:` base paths in `configs/*.yaml` while preserving each run-specific suffix.

Both scripts support:

- `--dry-run` to preview matches without modifying files.
- `--apply` to perform replacements.
- automatic timestamped backups under `.path_update_backups/` before any write.

Recommended workflow:

1. Run with `--dry-run`.
2. Run with `--apply`.
3. Review with `git diff`.

### Update workdir defaults in Slurm scripts

```bash
# Preview
scripts/update_workdir_roots.sh --dry-run \
  --old-workdir '/users/$USER/projects/ESFM' \
  --new-workdir '/path/to/repo/ESFM'

# Apply
scripts/update_workdir_roots.sh --apply \
  --old-workdir '/users/$USER/projects/ESFM' \
  --new-workdir '/path/to/repo/ESFM'
```

### Update log_dir base in config files

```bash
# Preview
scripts/update_logdir_roots.sh --dry-run \
  --old-log-base '/iopsstor/scratch/cscs/fozdemir/ESFM_outputs' \
  --new-log-base '/new/path/to/checkpoints/ESFM_outputs'

# Apply
scripts/update_logdir_roots.sh --apply \
  --old-log-base '/iopsstor/scratch/cscs/fozdemir/ESFM_outputs' \
  --new-log-base '/new/path/to/checkpoints/ESFM_outputs'
```

After applying updates, inspect the result:

```bash
git diff -- scripts/training scripts/inference configs
```

## Troubleshooting Notes

- If W&B logging is enabled, set `WANDB_KEY` (or disable with `--wnb_mode disabled`).
- Verify dataset paths in `dataset_config.yaml` and hard-coded path prefixes in entrypoint scripts.
- If running outside Slurm/CSCS, remove or adapt Slurm-specific environment and NCCL settings.
- Ensure CUDA/NCCL/PyTorch versions are compatible across all nodes for distributed runs.

## Citation

You can cite ESFM as follows:
```
@misc{ozdemir2026esfm,
      title={Earth System Foundation Model (ESFM): A unified framework for heterogeneous data integration and forecasting}, 
      author={Firat Ozdemir and Yun Cheng and Salman Mohebi and Fanny Lehmann and Simon Adamov and Zhenyi Zhang and Leonardo Trentini and Dana Grund and Oliver Fuhrer and Torsten Hoefler and Siddhartha Mishra and Sebastian Schemm and Benedikt Soja and Mathieu Salzmann},
      year={2026},
      eprint={2605.00850},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph},
      url={https://arxiv.org/abs/2605.00850}, 
}
```


## License

MIT License. See `LICENSE.txt`.

## Acknowledgements

This work was supported under project IDs a01 and a122 as part of the Swiss AI Initiative, through a grant from the ETH Domain and computational resources provided by the Swiss National Supercomputing Centre (CSCS) under the Alps infrastructure. 

## Contributing

Contributions are welcome.

