"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import contextlib
from datetime import timedelta

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from esfm.batch import Batch
from esfm.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    variables_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
    absolute_time_expansion_minutes,
)
from esfm.model.patchembed import LevelPatchEmbed, VariablePatchEmbed
from esfm.model.perceiver import MLP, PerceiverResampler
from esfm.model.posencoding import pos_scale_enc
from esfm.model.util import (
    check_lat_lon_dtype,
    init_weights,
)

__all__ = ["Perceiver3DEncoder"]

NUM_MAX_SURF_VARS=201
NUM_MAX_ATMOS_VARS=201
# All new variables should be added in the list before training for any new variables
# statistic variables included in surf variables
ATMOS_VARS = {'z': 0, 'u': 1, 'v': 2, 't': 3, 'q': 4, 'w': 5, 
              't_cosmo': 6, 'u_cosmo': 7, 'v_cosmo': 8, 'w_cosmo': 9, 'q_cosmo': 10, 
              'rh_cosmo': 11, 'pp_cosmo': 12, 
              '<N/A>': NUM_MAX_ATMOS_VARS - 1}
SURF_VARS = {'lsm': 0, 'z': 1, 'slt': 2, '2t': 3, '10u': 4, '10v': 5, 
             'msl': 6, '2q': 7, 'tp': 8, 'tp_mswep': 9, 'pe': 10, 
             'e': 11, 'r': 12, 'swvl': 13, 'sst': 14, 'ci': 15, 'co2': 16, 'tp_log': 17, 'tp_mswep_log': 18,
             'dt': 19, 'tp_6hr': 20, 'pe_6hr': 21, 'ssr_6hr': 22, 'ssrd_6hr': 23, 'ttr_6hr': 24, 'tsr_6hr': 25, 
             'e_6hr': 26, 'r_6hr': 27, 'sf_6hr': 28, 'str_6hr': 29, 'sshf_6hr': 30, 'slhf_6hr': 31, 'strd_6hr': 32, 
             'ir_mod': 33, 'nir_mod': 34, 'ir_myd': 35, 'nir_myd': 36, 'pa': 37, 'ws': 38, 'wd': 39, 
             'u_10m_cosmo': 40, 'v_10m_cosmo': 41, 'vmax_10m': 42, '2d': 43, 'sp': 44, 'clct_cosmo': 45,
             'alhfl_s': 46, 'ashfl_s': 47, 'asob_s': 48, 'athb_s': 49, 'tp_cosmo': 50, 'tcwv': 51,
             'w_snow_cosmo': 52, 'z_cosmo': 53, 'hsurf': 54, 'p0fl': 55, 'ts': 56,
             'tws': 57, 'tws_gou': 58, 'tws_itsg': 59, 'tws_jpl': 60,
             '<N/A>': NUM_MAX_SURF_VARS - 1}
             
def get_indices(variables, variables_registry):
    """ Returns list of integers for the variables based on their indices
        within variables_registry. """

    indices = []
    for var in variables:
        if var in variables_registry:
            indices.append(variables_registry[var])
        else:
            raise ValueError(f"Variable '{var}' not found.")
    return indices


def aggregate_levels(x: torch.Tensor, atmos_latents, level_agg, ) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        x = level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x
    
class Perceiver3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...] | None,
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.0,  
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
        stabilise_level_agg: bool = False,
        patch_tokenizer_identifier = None,
        **kwargs,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.1`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to 1e-5.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
        """
        super().__init__()

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_tokenizer_identifier = patch_tokenizer_identifier
        self.disable_flashattention = kwargs.get('disable_flashattention', False)
        self.absolute_time_embedding_in_minutes = kwargs.get('absolute_time_embedding_in_minutes', False)

        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        # Latent tokens
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        # Patch embeddings
        assert max_history_size > 0, "At least one history step is required."
        if self.patch_tokenizer_identifier is None:
            ## Use a single patch tokenizer for all input (grid resolutions)
            self.surf_token_embeds = LevelPatchEmbed(
                surf_vars,
                patch_size,
                embed_dim,
                max_history_size,
            )
            self.atmos_token_embeds = LevelPatchEmbed(
                atmos_vars,
                patch_size,
                embed_dim,
                max_history_size,
            )
        else:
            # Define multiple patch embeddings, to be decided based on the input grid resolution:
            self.surf_token_embeds = nn.ModuleDict()
            self.atmos_token_embeds = nn.ModuleDict()
            for key, patch_size in self.patch_tokenizer_identifier.d_patchsize.items():
                self.surf_token_embeds[key] = LevelPatchEmbed(
                    surf_vars,
                    patch_size,
                    embed_dim,
                    max_history_size,
                )
                
                self.atmos_token_embeds[key] = LevelPatchEmbed(
                    atmos_vars,
                    patch_size,
                    embed_dim,
                    max_history_size,
                )

        # Learnable pressure level aggregation
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
            disable_flashattention=self.disable_flashattention
        )

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)


    def forward(self, batch: Batch, lead_time: timedelta) -> torch.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.
            lead_time (timedelta): Lead time.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)
        x_static = torch.stack(tuple(batch.static_vars.values()), dim=2) if batch.has_static_vars else None # static variables are optional
        x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)
        
        if self.patch_tokenizer_identifier is None:
            patch_size = self.patch_size
            grid_resolution_str = None
            atmos_token_embeds = self.atmos_token_embeds
            surf_token_embeds = self.surf_token_embeds
        else:
            patch_size = self.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
            grid_resolution_str = self.patch_tokenizer_identifier.get_resolution_str(batch.metadata.grid_resolution)
            atmos_token_embeds = self.atmos_token_embeds[grid_resolution_str]
            surf_token_embeds = self.surf_token_embeds[grid_resolution_str]

        B, T, _, C, H, W = x_atmos.size()
        assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."

        # is static variables are given, they are concatenated with surface variables
        if len(static_vars):
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # Patch embed the surface level.
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w") ## x_surf includes the static variables
        x_surf = surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Patch embed the atmospheric levels.
        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        x_atmos = atmos_token_embeds(x_atmos, atmos_vars)
        x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf)) ##consider doing this under autocast. This is also optional.

        # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        x_atmos = aggregate_levels(x_atmos, self.atmos_latents, self.level_agg)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the amospheric levels.
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        if self.absolute_time_embedding_in_minutes:
            absolute_times_list = [t.timestamp() / 60 for t in batch.metadata.time]  # Times in minutes
            absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
            absolute_time_encode = absolute_time_expansion_minutes(absolute_times, self.embed_dim)
        else:
            absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
            absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
            absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x

class AxialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, heads_dim=None, dropout=0.0, attn_drop_rate=0.1, bias=True, disable_flashattention=False):        
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (embed_dim // num_heads) if heads_dim is None else heads_dim
        hidden_dim = self.head_dim * num_heads
        assert (embed_dim % num_heads == 0)
        self.attn_drop = attn_drop_rate
        self.disable_flashattention = disable_flashattention
        self.flash_attn_qkvpacked_func = None
        
        if not disable_flashattention:
            try:
                from flash_attn import flash_attn_qkvpacked_func  # type: ignore[import-not-found]
                self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
            except ImportError:
                print("Flash Attention is not installed. Falling back to regular attention.")
                self.disable_flashattention = True
        
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        
         # Linear layer for output projection
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.ln1 = nn.LayerNorm(embed_dim) #post mha
        self.ln2 = nn.LayerNorm(embed_dim) # post ff
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights to match MultiheadAttention defaults
        self._reset_parameters()

    def _reset_parameters(self):
        # Match torch.nn.MultiheadAttention's weight initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, kv = None):
        '''Expects input of shape: b=(batch_size x [Pressure_levels] x seq_len=tokens), l=#variables, h=#embed_dim'''
        kv = x if kv is None else kv
        q, k, v = (self.q_proj(x), self.k_proj(kv), self.v_proj(kv))
        attn_dropout = self.attn_drop if self.training else 0.0

        b, l, hidden_dim = q.shape
        # h = self.num_heads
        # e = self.head_dim
        q = q.view(b, l, self.num_heads, self.head_dim) # (batch, seqlen=#variables,  heads, head_dim)
        k = k.view(b, l, self.num_heads, self.head_dim)
        v = v.view(b, l, self.num_heads, self.head_dim)

        use_flash_attn = q.dtype in [torch.float16, torch.bfloat16]
        if self.disable_flashattention or self.flash_attn_qkvpacked_func is None:
            use_flash_attn = False
        if use_flash_attn:
            qkv = torch.stack((q, k, v), dim=2) # (batch, seqlen, 3, heads, head_dim)
            #flash_attn_qkvpacked_func expects input qkv: (batch_size, seqlen, 3, nheads, headdim), out: (batch_size, seqlen, nheads, headdim)
            b = q.shape[0]
            b_lim = 40_000  # Flash Attention has a lower tensor size limit that causes "RuntimeError: CUDA error: invalid configuration argument"
            if b < b_lim:
                attn_output = self.flash_attn_qkvpacked_func(qkv, dropout_p=attn_dropout)
            else:
                attn_output = torch.empty_like(qkv[:, :, 0, :, :])  
                for i in range(0, b, b_lim):
                    qkv_batch = qkv[i:i+b_lim]  
                    attn_output[i:i+b_lim] = self.flash_attn_qkvpacked_func(qkv_batch, dropout_p=attn_dropout)

        else:
            # normal attention
            q = rearrange(q, "b l h e -> b h l e")
            k = rearrange(k, "b l h e -> b h l e")
            v = rearrange(v, "b l h e -> b h l e")
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout)
            attn_output = rearrange(attn_output, "b h l e -> b l h e") # h: num_heads, e: head_dim -> h*e = hidden_dim
        
        attn_output = attn_output.reshape(b, l, hidden_dim)  # 

        attn_output = x + attn_output #residual # x: #embed_dim, attn_output: #hidden_dim
        attn_output = self.ln1(attn_output) #post-norm
        
        mlp_output = self.out_proj(attn_output) # hidden_dim -> embed_dim
        output = attn_output + self.dropout(mlp_output) #residual
        output = self.ln2(output) #post-norm
        
        return output
        
class Perceiver3DEncoderWithVariableAggregation(nn.Module):
    def __init__(
        self, 
        surf_vars: tuple[str, ...],
        static_vars: tuple[str, ...] | None,
        atmos_vars: tuple[str, ...],
        patch_size=4, 
        latent_levels: int = 8,
        embed_dim = 1024, 
        num_heads = 8, 
        head_dim = 32, 
        drop_rate = 0.1, 
        depth = 2, 
        mlp_ratio = 4, 
        max_history_size = 2, 
        perceiver_ln_eps = 0.00001, 
        stabilise_level_agg: bool = False,
        axial_attention=True,
        autocast: bool = False,
        patch_tokenizer_identifier = None,
        extensive_checkpointing: bool = True,
        absolute_time_embedding_in_minutes: bool = False,
        var_attn_chunk_size: int | None = None,
        *args, **kwargs
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_tokenizer_identifier = patch_tokenizer_identifier
        self.disable_flashattention = kwargs.get('disable_flashattention', False)
        self.extensive_checkpointing = extensive_checkpointing
        self.absolute_time_embedding_in_minutes = absolute_time_embedding_in_minutes
        self.var_attn_chunk_size = var_attn_chunk_size
        
        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars
        
        # Latent tokens
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))
        
        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)
        
        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)
        
        # Patch embeddings
        assert max_history_size > 0, "At least one history step is required."
        
        self.autocast = autocast
        self.dtype_dynamic = None # assigned to dtype of x_atmos in forward call
        # latent vector for atmospheric varialbes
        self.atmos_latents_vars = nn.Parameter(torch.randn(1, embed_dim))
        # latent vector for surface variables
        self.surf_latents_vars = nn.Parameter(torch.randn(1, embed_dim))
        # Learnable token for NaNs
        nan_token = nn.Parameter(torch.randn(embed_dim))

        self.atmos_variables_embed = nn.Linear(embed_dim, embed_dim)
        self.surf_variables_embed = nn.Linear(embed_dim, embed_dim)
        self.axial_attention = axial_attention
        
        # Add embedding layers for variables
        # UseNUM_MAX_SURF_VARS for surface variables
        self.surf_vars_embedding = nn.Embedding(NUM_MAX_SURF_VARS, embed_dim)
        # Use NUM_MAX_SURF_VARS for atmospheric variables
        self.atmos_vars_embedding = nn.Embedding(NUM_MAX_ATMOS_VARS, embed_dim)


        if self.axial_attention:
            self.atmos_var_attn = AxialAttention( 
                embed_dim=embed_dim,
                num_heads=num_heads,
                heads_dim=head_dim,
                dropout=0,
                disable_flashattention=self.disable_flashattention,
            )
            self.surf_var_attn = AxialAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                heads_dim=head_dim,
                dropout=0,
                disable_flashattention=self.disable_flashattention,
            )
             
        if self.patch_tokenizer_identifier is None:
            ## Use a single patch tokenizer for all input (grid resolutions)
            self.surf_token_embeds = VariablePatchEmbed(
                var_names=surf_vars+static_vars, ## bug here, should be surf_vars+static_vars --> surf_vars. Fix it after active training is over. 
                patch_size=patch_size,
                embed_dim=embed_dim,
                history_size=max_history_size,
                nan_token=nan_token,
            )
            
            self.atmos_token_embeds = VariablePatchEmbed(
                var_names=atmos_vars,
                patch_size=patch_size,
                embed_dim=embed_dim,
                history_size=max_history_size,
                nan_token=nan_token,
            )
        else:
            # Define multiple patch embeddings, to be decided based on the input grid resolution:
            self.surf_token_embeds = nn.ModuleDict()
            self.atmos_token_embeds = nn.ModuleDict()
            for key, patch_size in self.patch_tokenizer_identifier.d_patchsize.items():
                self.surf_token_embeds[key] = VariablePatchEmbed(
                    var_names=surf_vars+static_vars, ## bug here, should be surf_vars+static_vars --> surf_vars (static vars already added to surf vars). Fix it after active training is over. 
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    history_size=max_history_size,
                    nan_token=nan_token,
                )
                
                self.atmos_token_embeds[key] = VariablePatchEmbed(
                    var_names=atmos_vars,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    history_size=max_history_size,
                    nan_token=nan_token,
                )
                
        
        self.atmos_vars_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
        )

        self.surf_vars_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
        )
        
        # Learnable pressure level aggregation
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
            ln_k_q=stabilise_level_agg,
        )

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_vars(self, x: torch.Tensor, latents, perceiver_module) -> torch.Tensor:
        B, _, L, _ = x.shape
        latents = latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (V_A, D) to (B, V_A, L, D)

        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # (B * L, V_A, D)
        latents = torch.einsum("bvld->blvd", latents)
        latents = latents.flatten(0, 1)  # (B * L, V_A, D)

        x = perceiver_module(latents, x)  # (B * L, V, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, V, D)
        x = torch.einsum("blvd->bvld", x)  # (B, V, L, D)
        return x

    def _checkpointed_var_attn(self, module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        chunk_size = self.var_attn_chunk_size
        if chunk_size is not None and n > chunk_size:
            out_chunks = [
                checkpoint(module, x[i:i + chunk_size], use_reentrant=False)
                for i in range(0, n, chunk_size)
            ]
            return torch.cat(out_chunks, dim=0)
        return checkpoint(module, x, use_reentrant=False)

    def process_surf(self, x_surf, surf_vars, dtype, grid_resolution_str=None):
        # Patch embed the surface level.
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        if grid_resolution_str is None:
            surf_token_embeds = self.surf_token_embeds
        else:
            surf_token_embeds = self.surf_token_embeds[grid_resolution_str]
        x_surf = surf_token_embeds(x_surf, surf_vars) # (B, V, L, D)

        surf_vars_tensor = torch.tensor(
            get_indices(surf_vars, SURF_VARS), device=x_surf.device
        )
        #surf_vars_encode = variables_expansion(surf_vars_tensor, self.embed_dim).to(dtype=x_surf.dtype)
        surf_vars_encode = self.surf_vars_embedding(surf_vars_tensor)
        surf_vars_embed = self.surf_variables_embed(surf_vars_encode)[None, :, None, :] # (1, V, 1, D)
        x_surf = x_surf + surf_vars_embed
        b = x_surf.size(0)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            if self.axial_attention:
                x_surf = rearrange(x_surf, "b v l d -> (b l) v d", b=b)
                if self.extensive_checkpointing:
                    x_surf = self._checkpointed_var_attn(self.surf_var_attn, x_surf) # ( (B, L), V, D)
                else:
                    x_surf = self.surf_var_attn(x_surf)
                x_surf = rearrange(x_surf, '(b l) v d -> b v l d', b=b)
            
            x_surf = self.aggregate_vars(x_surf, self.surf_latents_vars, self.surf_vars_agg) # calls perceiver module
            
            # Add surface level encoding. This helps the model distinguish between surface and
            # atmospheric levels.
            x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
            # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
            x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))
        return x_surf
    
    def process_atmos(self, x_atmos, atmos_vars, B, C, dtype, grid_resolution_str=None):
        # Patch embed the atmospheric levels.
        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        if grid_resolution_str is None:
            atmos_token_embeds = self.atmos_token_embeds
        else:
            atmos_token_embeds = self.atmos_token_embeds[grid_resolution_str]
        x_atmos = atmos_token_embeds(x_atmos, atmos_vars) # (B*C, V, L=tokens, D)
        # x_atmos = rearrange(x_atmos, '(b c v) l w -> (b l) c v w', b=B, c=C)

        # Add variable encoding
        atmos_vars_tensor = torch.tensor(
            get_indices(atmos_vars, ATMOS_VARS), device=x_atmos.device
        )
        #atmos_vars_encode = variables_expansion(atmos_vars_tensor, self.embed_dim).to(dtype=dtype)
        atmos_vars_encode = self.atmos_vars_embedding(atmos_vars_tensor)
        atmos_vars_embed = self.atmos_variables_embed(atmos_vars_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_vars_embed
        dype_dynamic = dtype
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            
            # x_atmos = rearrange(x_atmos, 'B V L D -> (B L) V D')
            if self.axial_attention:
                x_atmos = rearrange(x_atmos, "(b c) v l d -> (b c l) v d", b=B, c=C)
                if self.extensive_checkpointing:
                    x_atmos = self._checkpointed_var_attn(self.atmos_var_attn, x_atmos) # ( (B, C, L), V, D)
                else:
                    x_atmos = self.atmos_var_attn(x_atmos) # ( (B, C, L), V, D)
                x_atmos = rearrange(x_atmos, '(B C L) V D -> (B C) V L D', B=B, C=C)
            x_atmos = self.aggregate_vars(x_atmos, self.atmos_latents_vars, self.atmos_vars_agg) # shape: ( (B C), V=1, L, D)
            
            x_atmos = rearrange(x_atmos, '(B C) 1 L D -> B C L D', C=C)
        
        
        if self.autocast: # cast back to original dtype.
            x_atmos = x_atmos.to(dtype=dype_dynamic)

        return x_atmos
        
    def forward(self, batch, lead_time):
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.
            lead_time (timedelta): Lead time.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2) if batch.has_surf_vars else None
        x_static = torch.stack(tuple(batch.static_vars.values()), dim=2) if batch.has_static_vars else None
        x_atmos = torch.stack(tuple(batch.atmos_vars.values()), dim=2)
        
        if self.patch_tokenizer_identifier is None:
            patch_size = self.patch_size
            grid_resolution_str = None
        else:
            patch_size = self.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
            grid_resolution_str = self.patch_tokenizer_identifier.get_resolution_str(batch.metadata.grid_resolution)
        

        dtype = x_atmos.dtype  # When using mixed precision, we need to keep track of the dtype.
        self.dtype_dynamic = dtype
        device = x_atmos.device
        B, T, _, C, H, W = x_atmos.size()

        if len(static_vars) == 0:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # embed surface variables & embed and aggregate atmos variables
        x_surf = self.process_surf(x_surf, surf_vars, dtype, grid_resolution_str) # (B, C=1, L, D)
        x_atmos = self.process_atmos(x_atmos, atmos_vars, B, C, dtype, grid_resolution_str) # (B, C_A, L, D)
            
            
        # # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        atmos_levels_tensor = torch.tensor(atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)
        
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():
            # # Aggregate over pressure levels.
            x_atmos = aggregate_levels(x_atmos, self.atmos_latents, self.level_agg)  # (B, C_A, L, D) to (B, C, L, D)
        
        # Concatenate the surface level with the amospheric levels.
        x = torch.cat((x_surf, x_atmos), dim=1) if len(surf_vars) > 0 else x_atmos
        
        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        if self.autocast: #cast back to original dtype
            x = x.to(dtype=self.dtype_dynamic)
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        if self.absolute_time_embedding_in_minutes:
            absolute_times_list = [t.timestamp() / 60 for t in batch.metadata.time]  # Times in minutes
            absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
            absolute_time_encode = absolute_time_expansion_minutes(absolute_times, self.embed_dim)
        else:
            absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
            absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
            absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x
