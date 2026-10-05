"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from esfm.batch import Batch
from esfm.model.encoder import Perceiver3DEncoder, Perceiver3DEncoderWithVariableAggregation
from esfm.model.lora import LoRAMode
from esfm.model.swin3d import BasicLayer3D


__all__ = ["ESFMEncoder"]

class ResolutionSpecificPatchTokenizers:
    def __init__(self, ):
        self.resolutions_str = ['very_coarse', 'coarse', 'medium', 'fine', 'very_fine', 'station']
        self.num_resolutions = len(self.resolutions_str)
        self.patch_sizes = [4, 4, 4, 4, 4, 1]
        self.resolutions_degree = [1.5, 0.50, 0.15, 0.05, 0.00001]
        self.d_patchsize = dict(zip(self.resolutions_str, self.patch_sizes))
        if self.num_resolutions != len(self.resolutions_degree) + 1:
            raise ValueError('Number of resolutions and degrees do not match')
        if not all(self.resolutions_degree[i] > self.resolutions_degree[i + 1] for i in range(len(self.resolutions_degree) - 1)):
            raise ValueError('Resolutions are not in decreasing order')
    def get_resolution_str(self, grid_resolution):
        if grid_resolution > self.resolutions_degree[0]: #(1.5deg, inf)
            return self.resolutions_str[0]
        elif grid_resolution > self.resolutions_degree[1]: #(0.5, 1.5]
            return self.resolutions_str[1]
        elif grid_resolution > self.resolutions_degree[2]: #(0.15, 0.5]
            return self.resolutions_str[2]
        elif grid_resolution > self.resolutions_degree[3]:  #(0.05, 0.15]
            return self.resolutions_str[3]
        elif grid_resolution > self.resolutions_degree[4]:  #(1e-5, 0.05]
            return self.resolutions_str[4]
        else:  # (0, 1e-5] # ~10m2 --> station
            return self.resolutions_str[5]
    def get_patch_size(self, grid_resolution):
        return self.patch_sizes[self.resolutions_str.index(self.get_resolution_str(grid_resolution))]

class ESFMEncoder(torch.nn.Module):
    """The ESFM model.

    Defaults to to the 1.3 B parameter configuration.
    """

    def __init__(
        self,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 32,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        use_lora: bool = True,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        autocast: bool = False,
        num_ensemble: int = 1,  # Number of ensemble members
        variable_aggregation=False,
        axial_attention=True,
        use_resolution_specific_patch_tokenizers: bool = False,
        disable_flashattention: bool = False,
        add_qk_norm_to_swin3d: bool = False,
        token_aggregation_attn_drop_rate: float = 0.1,
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model.
            static_vars (tuple[str, ...], optional): All static variables supported by the
                model.
            atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model.
            window_size (tuple[int, int, int], optional): Vertical height, height, and width of the
                window of the underlying Swin transformer.
            encoder_depths (tuple[int, ...], optional): Number of blocks in each encoder layer.
            encoder_num_heads (tuple[int, ...], optional): Number of attention heads in each encoder
                layer. The dimensionality doubles after every layer. To keep the dimensionality of
                every head constant, you want to double the number of heads after every layer. The
                dimensionality of attention head of the first layer is determined by `embed_dim`
                divided by the value here. For all cases except one, this is equal to `64`.
            decoder_depths (tuple[int, ...], optional): Number of blocks in each decoder layer.
                Generally, you want this to be the reversal of `encoder_depths`.
            decoder_num_heads (tuple[int, ...], optional): Number of attention heads in each decoder
                layer. Generally, you want this to be the reversal of `encoder_num_heads`.
            latent_levels (int, optional): Number of latent pressure levels.
            patch_size (int, optional): Patch size.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. The dimensionality of these attention heads will be equal to
                `embed_dim` divided by this value.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            token_aggregation_attn_drop_rate (float, optional): Attention dropout for encoder
                variable and pressure-level aggregation. Defaults to `0.1`.
            drop_path (float, optional): Drop-path rate.
            enc_depth (int, optional): Number of Perceiver blocks in the encoder.
            dec_depth (int, optioanl): Number of Perceiver blocks in the decoder.
            dec_mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs in the
                decoder. The embedding dimensionality here is different, which is why this is a
                separate parameter.
            perceiver_ln_eps (float, optional): Epsilon in the perceiver layer norm. layers. Used
                to stabilise the model.
            max_history_size (int, optional): Maximum number of history steps. You can load
                checkpoints with a smaller `max_history_size`, but you cannot load checkpoints
                with a larger `max_history_size`.
            timestep (timedelta, optional): Timestep of the model. Defaults to 6 hours.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            use_lora (bool, optional): Use LoRA adaptation.
            lora_steps (int, optional): Use different LoRA adaptation for the first so-many roll-out
                steps.
            lora_mode (str, optional): LoRA mode. `"single"` uses the same LoRA for all roll-out
                steps from the second step onward, and `"all"` uses a different LoRA for every
                roll-out step from the second step onward. LoRA is zeroed out at step 0 in both
                modes. Defaults to `"single"`.
            surf_stats (dict[str, tuple[float, float]], optional): For these surface-level
                variables, adjust the normalisation to the given tuple consisting of a new location
                and scale.
            autocast (bool, optional): Use `torch.autocast` to reduce memory usage. Defaults to
                `False`.
            variable_aggregation (bool, optional): specify if we want to use self-attention mechanism
                across variables and them apply variable aggregation
            axial_attention (bool, optional): Use axial attention in Perceiver blocks.
            use_resolution_specific_patch_tokenizers (bool, optional): Use different patch tokenizers for different resolutions.
            disable_flashattention (bool, optional): Disable flash attention.
            add_qk_norm_to_swin3d (bool, optional): Add layer norm layers to Q & K in the Swin3D transformer blocks of the backbone.
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.autocast = autocast
        self.max_history_size = max_history_size
        self.timestep = timestep
        self.axial_attention = axial_attention
        self.variable_aggregation = variable_aggregation
        self.use_resolution_specific_patch_tokenizers = use_resolution_specific_patch_tokenizers
        self.disable_flashattention = disable_flashattention
        if self.use_resolution_specific_patch_tokenizers:
            self.patch_tokenizer_identifier = ResolutionSpecificPatchTokenizers()
        else:
            self.patch_tokenizer_identifier = None

        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually "
                f"adjusted: {', '.join(sorted(self.surf_stats.keys()))}. "
                f"Please ensure that this is right!",
                stacklevel=2,
            )

        encoder = Perceiver3DEncoderWithVariableAggregation if variable_aggregation else Perceiver3DEncoder 
        self.encoder = encoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            token_aggregation_attn_drop_rate=token_aggregation_attn_drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=head_dim,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
            axial_attention=axial_attention,
            autocast=autocast,
            patch_tokenizer_identifier=self.patch_tokenizer_identifier,
            disable_flashattention=disable_flashattention,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        if self.use_resolution_specific_patch_tokenizers:
            patch_size = self.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
        else:
            patch_size = self.patch_size
        batch = batch.crop(patch_size=patch_size)
        batch = batch.to(p.device)
        is_global_observation = batch.metadata.is_global_observation

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // patch_size,
            W // patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = batch.batch_and_history_dims

        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        x = self.encoder(
            batch,
            lead_time=self.timestep,
        )
        
        return x

    def load_checkpoint(self, repo: str, name: str, strict: bool = True) -> None:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str): Name of the repository of the form `user/repo`.
            name (str): Path to the checkpoint relative to the root of the repository, e.g.
                `checkpoint.cpkt`.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict)

    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device

        d = torch.load(path, map_location=device, weights_only=True)

        # You can safely ignore all cumbersome processing below. We modified the model after we
        # trained it. The code below manually adapts the checkpoints, so the checkpoints are
        # compatible with the new model.

        # Remove possibly prefix from the keys.
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v

        # Convert the ID-based parametrization to a name-based parametrization.
        if "encoder.surf_token_embeds.weight" in d:
            weight = d["encoder.surf_token_embeds.weight"]
            del d["encoder.surf_token_embeds.weight"]
            bias = d["encoder.surf_token_embeds.bias"]
            if self.variable_aggregation or self.use_resolution_specific_patch_tokenizers:
                del d["encoder.surf_token_embeds.bias"]

            assert weight.shape[1] == 4 + 3
            if self.use_resolution_specific_patch_tokenizers and not self.variable_aggregation:
                for k in self.patch_tokenizer_identifier.resolutions_str:
                    d[f"encoder.surf_token_embeds.{k}.bias"] = bias
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                if self.use_resolution_specific_patch_tokenizers:
                    for k in self.patch_tokenizer_identifier.resolutions_str:
                        d[f"encoder.surf_token_embeds.{k}.weights.{name}"] = weight[:, [i]]
                        if self.variable_aggregation:
                            d[f"encoder.surf_token_embeds.{k}.bias.{name}"] = bias ##duplicate pretrained bias to all vars separately.
                else:
                    d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]
                    if self.variable_aggregation:
                        d[f"encoder.surf_token_embeds.bias.{name}"] = bias ##duplicate pretrained bias to all vars separately.

        if "encoder.atmos_token_embeds.weight" in d:
            weight = d["encoder.atmos_token_embeds.weight"]
            del d["encoder.atmos_token_embeds.weight"]
            bias = d["encoder.atmos_token_embeds.bias"] 
            if self.variable_aggregation or self.use_resolution_specific_patch_tokenizers:
                del d["encoder.atmos_token_embeds.bias"]

            assert weight.shape[1] == 5
            if self.use_resolution_specific_patch_tokenizers and not self.variable_aggregation:
                for k in self.patch_tokenizer_identifier.resolutions_str:
                    d[f"encoder.atmos_token_embeds.{k}.bias"] = bias
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if self.use_resolution_specific_patch_tokenizers:
                    for k in self.patch_tokenizer_identifier.resolutions_str:
                        d[f"encoder.atmos_token_embeds.{k}.weights.{name}"] = weight[:, [i]]
                        if self.variable_aggregation:
                            d[f"encoder.atmos_token_embeds.{k}.bias.{name}"] = bias
                else:
                    d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]
                    if self.variable_aggregation:
                        d[f"encoder.atmos_token_embeds.bias.{name}"] = bias


        # Check if the history size is compatible and adjust weights if necessary.
        if self.use_resolution_specific_patch_tokenizers:
            current_history_size = d["encoder.surf_token_embeds.medium.weights.2t"].shape[2]
        else:
            current_history_size = d["encoder.surf_token_embeds.weights.2t"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} "
                f"into model with `max_history_size` {self.max_history_size}."
            )
            
        load_result = self.load_state_dict(d, strict=strict)

        # Log any missing or unexpected keys
        if load_result.missing_keys:
            print("Missing keys when loading checkpoint:")
            for key in load_result.missing_keys:
                print(f"  {key}")
        
        if load_result.unexpected_keys:
            print("Unexpected keys in checkpoint:")
            for key in load_result.unexpected_keys:
                print(f"  {key}")

    def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
        `max_history_size` than the current model.

        If a checkpoint was trained with a larger `max_history_size` than the current model,
        this function will assert fail to prevent loading the checkpoint. This is to
        prevent loading a checkpoint which will likely cause the checkpoint to degrade is
        performance.

        This implementation copies weights from the checkpoint to the model and fills zeros
        for the new history width dimension. It mutates `checkpoint`.
        """
        for name, weight in list(checkpoint.items()):
            # We only need to adapt the patch embedding in the encoder.
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                # This shouldn't get called with current logic but leaving here for future proofing
                # and in cases where its called outside current context.
                if not (weight.shape[2] <= self.max_history_size):
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
                        f"into model with `max_history_size` {self.max_history_size}."
                    )

                # Initialize the new weight tensor.
                new_weight = torch.zeros(
                    (weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                # Copy the existing weights to the new tensor by duplicating the histories provided
                # into any new history dimensions. The rest remains at zero.
                new_weight[:, :, : weight.shape[2]] = weight

                checkpoint[name] = new_weight

    def configure_activation_checkpointing(self):
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.
        """
        apply_activation_checkpointing(self, check_fn=lambda x: isinstance(x, BasicLayer3D))
