"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

from datetime import timedelta

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from esfm.batch import Batch, Metadata
from esfm.model.fourier import levels_expansion
from esfm.model.perceiver import PerceiverResampler
from esfm.model.util import (
    check_lat_lon_dtype,
    init_weights,
    unpatchify,
)
from esfm.model.film import AdaptiveLayerNorm

__all__ = ["Perceiver3DDecoder"]


def get_2d_sincos_pos_embed_pytorch(embed_dim, grid_h, grid_w, device):
    """Generate 2D sine-cosine positional embeddings.
    Returns a tensor of shape [grid_h*grid_w, embed_dim].
    """
    assert embed_dim % 4 == 0  # 2 for sin/cos, 2 for h/w
    
    # 1. Create grid coordinates
    grid_h_coords = torch.arange(grid_h, device=device, dtype=torch.float32)
    grid_w_coords = torch.arange(grid_w, device=device, dtype=torch.float32)
    
    # For global weather data: Longitude periodic wrap (optional but recommended)
    # grid_w_coords = grid_w_coords * (2 * torch.pi / grid_w) 

    grid_h_mesh, grid_w_mesh = torch.meshgrid(grid_h_coords, grid_w_coords, indexing='ij')
    
    # 2. Compute omega (frequencies)
    # We use embed_dim // 4 because we have (sin_h, cos_h, sin_w, cos_w)
    dim_per_coord = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, dim_per_coord, 2, device=device).float() / dim_per_coord))
    
    # 3. Compute 1D embeddings for each axis
    # Outer product: [H*W, 1] @ [1, D/4] -> [H*W, D/4]
    out_h = torch.einsum('m,d->md', grid_h_mesh.reshape(-1), omega)
    out_w = torch.einsum('m,d->md', grid_w_mesh.reshape(-1), omega)
    
    pos_embed = torch.cat([
        torch.sin(out_h), torch.cos(out_h),
        torch.sin(out_w), torch.cos(out_w)
    ], dim=1) # [H*W, Embed_Dim]
    
    return pos_embed

class Perceiver3DDecoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-5,
        num_ensemble: int = 1,  # Number of ensemble members
        patch_tokenizer_identifier=None, 
        disable_flashattention: bool = False,
        add_token_pos_embedding: bool = False,
        num_max_ensembles: int = 1000,
        adaln_scale_bias: float = 1.0,
        extensive_checkpointing: bool = True,
        level_decoder_chunk_size: int | None = None,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            embed_dim (int, optional): Embedding dim.. Defaults to `1024`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `1`.
            head_dim (int, optional): Dimension of the attention heads used in the aggregation
                blocks. Defaults to `64`.
            num_heads (int, optional): Number of attention heads used in the aggregation blocks.
                Defaults to `8`.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimensionality.
                Defaults to `4.0`.
            drop_rate (float, optional): Drop-out rate for input patches. Defaults to `0.0`.
            perceiver_ln_eps (float, optional): Layer norm. epsilon for the Perceiver blocks.
                Defaults to `1e-5`.
            add_token_pos_embedding: Whether to add token positional embeddings. Defaults to `False`.
            num_max_ensembles: Maximum number of ensembles for embedding layer. Defaults to `1000`.
            adaln_scale_bias: Scale bias for the adaptive layer normalisation. Defaults to `1.0`.
            level_decoder_chunk_size: Maximum number of flattened `(B * L)` items passed to
                `level_decoder` in a single call. Larger grids are split into chunks along this
                batch dimension (each item is processed independently, so this is exact, not an
                approximation) to bound peak activation memory. `None` disables chunking.
                Defaults to None.
        """
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.embed_dim = embed_dim
        self.num_ensemble = num_ensemble
        self.patch_tokenizer_identifier = patch_tokenizer_identifier
        self.add_token_pos_embedding = add_token_pos_embedding
        self.num_max_ensembles = num_max_ensembles
        self.extensive_checkpointing = extensive_checkpointing
        self.level_decoder_chunk_size = level_decoder_chunk_size
        if self.add_token_pos_embedding:
            self.cache_pos_embeddings = {} ## cache for pos embeddings for different grid sizes

        self.level_decoder = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            residual_latent=True,
            ln_eps=perceiver_ln_eps,
            disable_flashattention=disable_flashattention,
        )
        
        if self.num_ensemble > 1: 
            self.ensemble_cond_mlp_surf = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )
            self.ensemble_embedding_surf = nn.Embedding(num_max_ensembles, embed_dim)
            self.ensemble_adaln_surf = AdaptiveLayerNorm(embed_dim, embed_dim, scale_bias=adaln_scale_bias) # adaLN-Zero
            self.ensemble_cond_mlp_atmos = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),
            )
            self.ensemble_embedding_atmos = nn.Embedding(num_max_ensembles, embed_dim)
            self.ensemble_adaln_atmos = AdaptiveLayerNorm(embed_dim, embed_dim, scale_bias=adaln_scale_bias) # adaLN-Zero
        else:
            self.ensemble_cond_mlp_surf = nn.Identity()
            self.ensemble_embedding_surf = nn.Identity()
            self.ensemble_adaln_surf = nn.Identity()
            self.ensemble_cond_mlp_atmos = nn.Identity()
            self.ensemble_embedding_atmos = nn.Identity()
            self.ensemble_adaln_atmos = nn.Identity()
        
        # Create ensemble of heads for each variable
        self.surf_heads = nn.ModuleDict()
        self.atmos_heads = nn.ModuleDict()
        
        if self.patch_tokenizer_identifier is None:
            for name in surf_vars:
                self.surf_heads[name] = nn.ModuleList([
                    nn.Linear(embed_dim, patch_size**2) 
                    for _ in range(1)
                ])

            for name in atmos_vars:
                self.atmos_heads[name] = nn.ModuleList([
                    nn.Linear(embed_dim, patch_size**2)
                    for _ in range(1)
                ])
        else:
            # Create heads for each grid size input
            for key, patch_size in self.patch_tokenizer_identifier.d_patchsize.items():
                self.surf_heads[key] = nn.ModuleDict()
                self.atmos_heads[key] = nn.ModuleDict()

                for name in surf_vars:
                    self.surf_heads[key][name] = nn.ModuleList([
                        nn.Linear(embed_dim, patch_size**2) 
                        for _ in range(1)
                    ])

                for name in atmos_vars:
                    self.atmos_heads[key][name] = nn.ModuleList([
                        nn.Linear(embed_dim, patch_size**2)
                        for _ in range(1)
                    ])


        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)
        self.apply(init_weights)

    def deaggregate_levels(self, level_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Deaggregate pressure level information.

        Args:
            level_embed (torch.Tensor): Level embedding of shape `(B, L, C, D)`.
            x (torch.Tensor): Aggregated input of shape `(B, L, C', D)`.

        Returns:
            torch.Tensor: Deaggregate output of shape `(B, L, C, D)`.
        """
        B, L, C, D = level_embed.shape
        level_embed = level_embed.flatten(0, 1)  # (BxL, C, D)
        x = x.flatten(0, 1)  # (BxL, C', D)
        _msg = f"Batch size mismatch. Found {level_embed.size(0)} and {x.size(0)}."
        assert level_embed.size(0) == x.size(0), _msg
        assert len(level_embed.shape) == 3, f"Expected 3 dims, found {level_embed.dims()}."
        assert x.dim() == 3, f"Expected 3 dims, found {x.dim()}."

        n = level_embed.size(0)
        chunk_size = self.level_decoder_chunk_size
        if self.extensive_checkpointing:
            if chunk_size is not None and n > chunk_size:
                out_chunks = []
                for i in range(0, n, chunk_size):
                    out_chunks.append(
                        checkpoint(
                            self.level_decoder,
                            level_embed[i:i + chunk_size],
                            x[i:i + chunk_size],
                            use_reentrant=False,
                        )
                    )
                x = torch.cat(out_chunks, dim=0)  # (BxL, C, D)
            else:
                x = checkpoint(self.level_decoder, level_embed, x, use_reentrant=False)  # (BxL, C, D)
        else:
            x = self.level_decoder(level_embed, x)  # (BxL, C, D)
        x = x.reshape(B, L, C, D)
        return x

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        patch_res: tuple[int, int, int],
        lead_time: timedelta,
    ) -> tuple[Batch, Batch, Batch]:  # Returns (mean_batch, std_batch, all_preds_batch)
        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        B, L, D = x.shape
        lat, lon = batch.metadata.lat, batch.metadata.lon
        dataset_name = batch.metadata.dataset_name
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]
        atmos_vars_output = batch.metadata.atmos_vars_output
        surf_vars_output = batch.metadata.surf_vars_output
        atmos_levels_output = batch.metadata.atmos_levels_output
        if atmos_vars_output is None:
            atmos_vars_output = atmos_vars
        if surf_vars_output is None:
            surf_vars_output = surf_vars
        if atmos_levels_output is None:
            atmos_levels_output = atmos_levels

        if self.patch_tokenizer_identifier is None:
            patch_size = self.patch_size
            grid_resolution_str = None
            surf_heads = self.surf_heads
            atmos_heads = self.atmos_heads
        else:
            patch_size = self.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
            grid_resolution_str = self.patch_tokenizer_identifier.get_resolution_str(batch.metadata.grid_resolution)
            surf_heads = self.surf_heads[grid_resolution_str]
            atmos_heads = self.atmos_heads[grid_resolution_str]
            
        # Create ensemble conditioning vector
        ## Randomly sample self.num_ensemble integers between 0 and self.num_max_ensembles-1 for each batch element. This allows training self.num_max_ensembles ensembles.
        ensemble_indices = torch.randperm(self.num_max_ensembles, device=x.device)[:self.num_ensemble].sort()[0]
        # ensemble_indices = torch.arange(self.num_ensemble, device=x.device) 
        ensemble_embed_surf_ = self.ensemble_embedding_surf(ensemble_indices)  # (E, D)
        ensemble_embed_surf = self.ensemble_cond_mlp_surf(ensemble_embed_surf_)  # (E, D)
        ensemble_embed_atmos_ = self.ensemble_embedding_atmos(ensemble_indices)  # (E, D)
        ensemble_embed_atmos = self.ensemble_cond_mlp_atmos(ensemble_embed_atmos_)  # (E, D)

            
        # Unwrap the latent level dimension.
        x = rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0], # encoder latent levels --> 4
            H=patch_res[1], # patches H 
            W=patch_res[2], # patches W
        )
        
        if self.add_token_pos_embedding:
            key_pos_emb = (patch_res[1], patch_res[2]) # (H, W)
            if key_pos_emb not in self.cache_pos_embeddings:
                pos_embedding = get_2d_sincos_pos_embed_pytorch(
                    self.embed_dim,
                    grid_h=patch_res[1],
                    grid_w=patch_res[2],
                    device=x.device
                ) # (H*W, D)
                self.cache_pos_embeddings[key_pos_emb] = pos_embedding
            else:
                pos_embedding = self.cache_pos_embeddings[key_pos_emb]
            pos_embedding = pos_embedding.unsqueeze(0).unsqueeze(2) # (1, H*W, 1, D)
            x = x + pos_embedding # broadcast add

        if len(surf_vars_output) > 0:
            surf_preds_ensemble = []
            for i_ens in range(self.num_ensemble):
                x_ = x[...,:1,:] # (B, (HW), C=1, D)
                if self.num_ensemble > 1: # use FiLM layer only if num_ensemble > 1
                    x_ = x[...,:1,:].reshape(x.size(0), -1, x.size(-1)) # (B, (HW), C=1, D) -- >(B, L=(HWC), D)
                    cond_ens = ensemble_embed_surf[i_ens].unsqueeze(0).expand(B, -1) # (D,) --> (B, D)
                    x_ens = x_ + self.ensemble_adaln_surf(x_,cond_ens) # (B, L=HWC, D)
                    x_ = x_ens.reshape(B, patch_res[1]*patch_res[2], 1, self.embed_dim) # (B, (HW), C=1, D)
                x_ens = torch.stack([
                    surf_heads[name][0](x_)  # [B, H*W, 1, patch_size**2]
                    for name in surf_vars_output
                ], dim=-1) # shape: [B, H*W, 1, patch_size**2, num_vars]
                x_ens = x_ens.reshape(*x_ens.shape[:3], -1) # [B, H*W, 1, patch_size**2 * num_vars]
                surf_preds_ens = unpatchify(x_ens, len(surf_vars_output), H, W, patch_size) # [B, L=HW, C=1, P*P*V_S] --> [B, V_S, C=1, H, W]
                surf_preds_ens = surf_preds_ens.squeeze(2) # [B, V_S, H, W]
                surf_preds_ensemble.append(surf_preds_ens)
            surf_preds_all = torch.stack(surf_preds_ensemble, dim=1)  # [B, E, V_S, H, W]


        # Process atmospheric variables
        atmos_levels_encode = levels_expansion(
            torch.tensor(atmos_levels_output, device=x.device), self.embed_dim
        ).to(dtype=x.dtype)
        levels_embed = self.atmos_levels_embed(atmos_levels_encode)
        levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
        x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :]) # shape: B (H W) C=3 D --> [B, L=HW, C=[13], D]

        # # Stack ensemble predictions
        atmos_preds_ensemble = []
        for i_ens in range(self.num_ensemble):
            x_atmos_ = x_atmos
            if self.num_ensemble > 1: # use FiLM layer only if num_ensemble > 1
                cond_ens = ensemble_embed_atmos[i_ens].unsqueeze(0).expand(B, -1) # (D,) --> (B, D)
                x_atmos_ = x_atmos.reshape(x_atmos.size(0), -1, x_atmos.size(-1)) # [B, L=HW, C=[13], D] --> (B, (HW), C=levels, D) -- >(B, L=(HWC), D)
                x_atmos_ens = x_atmos_ + self.ensemble_adaln_atmos(x_atmos_,cond_ens) # (B, L=HWC, D)
                x_atmos_ = x_atmos_ens.reshape(B, patch_res[1]*patch_res[2], x_atmos.size(2), self.embed_dim) # (B, (HW), C=levels, D)
            x_atmos_ens = torch.stack([
                atmos_heads[name][0](x_atmos_)  # [B, H*W, levels, patch_size**2]
                for name in atmos_vars_output
            ], dim=-1) # shape: [B, H*W, levels, patch_size**2, num_vars]
            x_atmos_ens = x_atmos_ens.reshape(*x_atmos_ens.shape[:3], -1) # [B, H*W, levels, patch_size**2 * num_vars]
            atmos_preds_ens = unpatchify(x_atmos_ens, len(atmos_vars_output), H, W, patch_size) # [B, V_A, levels, H, W]
            atmos_preds_ensemble.append(atmos_preds_ens)
        atmos_preds_all = torch.stack(atmos_preds_ensemble, dim=1)  # [B, E, V_A, levels, H, W]

        all_preds_batch = Batch(
            {v: surf_preds_all[:, :, i] for i, v in enumerate(surf_vars_output)},
            batch.static_vars,
            {v: atmos_preds_all[:, :, i] for i, v in enumerate(atmos_vars_output)},
            Metadata(
                dataset_name=dataset_name,
                lat=lat,
                lon=lon,
                time=tuple(t + lead_time for t in batch.metadata.time),
                atmos_levels=atmos_levels,
                locations=batch.metadata.locations,
                scales=batch.metadata.scales,
                grid_resolution=batch.metadata.grid_resolution,
                is_global_observation=batch.metadata.is_global_observation,
                rollout_step=batch.metadata.rollout_step + 1,
                atmos_vars_output=atmos_vars_output,
                surf_vars_output=surf_vars_output,
                atmos_levels_output=atmos_levels_output,
                lead_time_seconds=batch.metadata.lead_time_seconds,
            ),
        )
        
        return all_preds_batch
