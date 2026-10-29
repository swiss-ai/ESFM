# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.
"""Compare a trained ESFM checkpoint with explicit short-sequence attention (new) against the
flash-attention path it was trained with (old), and time both.

The old path is selected by patching `perceiver.SHORT_SEQUENCE_MAX` to 0, so both runs use the same
weights, batches and code otherwise. Reports, per sample:
  - one-step relative L2 difference new vs old, per variable (and old vs old as the noise floor),
  - RMSE against the target for both,
  - the relative difference over a short rollout,
and the forward (no grad) and forward+backward step time and peak memory of both paths.

Usage (single GPU, inside the container):
    python scripts/compare_short_seq_attention.py --config configs/config_ESFM_s_nm_ft.yaml \
        --ckpt_path /path/to/model_ckpt-step=...ckpt --n_samples 4 --rollout_steps 8
"""
import argparse
import contextlib
import os
import sys
import time
from unittest import mock

own_parser = argparse.ArgumentParser(add_help=False)
own_parser.add_argument("--ckpt_path", required=True)
own_parser.add_argument("--n_samples", type=int, default=4)
own_parser.add_argument("--rollout_steps", type=int, default=8)
own_parser.add_argument("--bench_iters", type=int, default=10)
own_parser.add_argument("--skip_train_bench", action="store_true")
own_args, remaining = own_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining + ["--use_inds_subset"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import inference  # noqa: E402  (parses --config etc.)
from esfm import ESFM  # noqa: E402
from esfm.model import perceiver  # noqa: E402
from esfm.rollout import rollout  # noqa: E402

args = inference.args
args.devices = 1  # no process group
args.num_workers = 2

ARCH = {
    "small": dict(encoder_depths=(2, 6, 2), encoder_num_heads=(4, 8, 16), decoder_depths=(2, 6, 2),
                  decoder_num_heads=(16, 8, 4), embed_dim=256, num_heads=8),
    "large": dict(encoder_depths=(6, 10, 8), encoder_num_heads=(8, 16, 32), decoder_depths=(8, 10, 6),
                  decoder_num_heads=(32, 16, 8), embed_dim=512, num_heads=16),
}


def old_path():
    """Disable the explicit short-sequence attention (flash / SDPA as before)."""
    return mock.patch.object(perceiver, "SHORT_SEQUENCE_MAX", 0)


def build_model(surf_vars, atmos_vars, static_vars):
    model = ESFM(
        use_lora=False,
        autocast=True,
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        drop_path=0.0,
        num_ensemble=args.num_ensemble,
        variable_aggregation=args.variable_aggregation,
        use_resolution_specific_patch_tokenizers=args.use_resolution_specific_patch_tokenizers,
        disable_flashattention=args.disable_flashattention,
        stabilise_level_agg=args.stabilise_level_agg,
        add_qk_norm_to_swin3d=args.add_qk_norm_to_swin3d,
        fix_qk_norm_in_swin3d=args.fix_qk_norm_in_swin3d,
        encoder_activation_checkpointing=args.act_checkpointing_encoder,
        decoder_activation_checkpointing=args.act_checkpointing_decoder,
        var_attn_chunk_size=args.var_attn_chunk_size if args.var_attn_chunk_size > 0 else None,
        level_decoder_chunk_size=args.level_decoder_chunk_size if args.level_decoder_chunk_size > 0 else None,
        absolute_time_embedding_in_minutes=args.absolute_time_embedding_in_minutes,
        add_token_pos_embedding_in_decoder=args.add_token_pos_embedding_in_decoder,
        num_max_ensembles=args.num_max_ensembles,
        ensemble_adaln_scale_bias=args.ensemble_adaln_scale_bias,
        **ARCH[args.str_architecture_size],
    )
    ckpt = torch.load(own_args.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded {own_args.ckpt_path} (global step {ckpt.get('global_step')})")
    return model.cuda()


def drop_unknown_vars(batch, surf_vars, atmos_vars, static_vars):
    for d, known in ((batch.static_vars, static_vars), (batch.surf_vars, surf_vars), (batch.atmos_vars, atmos_vars)):
        for k in [k for k in d if k not in known]:
            del d[k]
    return batch


def prepare(model, batch):
    p = next(model.parameters())
    patch_size = (model.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
                  if model.use_resolution_specific_patch_tokenizers else model.patch_size)
    return batch.type(p.dtype).crop(patch_size).to(p.device)


def all_vars(b):
    return {**{f"surf/{k}": v for k, v in b.surf_vars.items()}, **{f"atmos/{k}": v for k, v in b.atmos_vars.items()}}


def rel_diff(a, b):
    a, b = a.float(), b.float()
    m = torch.isfinite(a) & torch.isfinite(b)
    return ((a[m] - b[m]).norm() / b[m].norm().clamp_min(1e-30)).item()


def rmse(pred, target):
    pred, target = pred.float(), target.float().to(pred.device)
    if pred.shape != target.shape:
        return float("nan")
    m = torch.isfinite(pred) & torch.isfinite(target)
    return (pred[m] - target[m]).square().mean().sqrt().item()


@torch.no_grad()
def predict(model, batch_cpu):
    pred, _, _ = model(prepare(model, batch_cpu))
    return all_vars(pred)


@torch.no_grad()
def predict_rollout(model, batch_cpu, steps):
    return [all_vars(p.to("cpu")) for p in rollout(model, batch_cpu, steps)]


def time_step(model, batch_cpu, train):
    model.train(train)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(own_args.bench_iters + 3):
        batch = prepare(model, batch_cpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with contextlib.nullcontext() if train else torch.no_grad():
            _, _, preds = model(batch)
            if train:
                loss = sum(v.float().nan_to_num().mean() for v in all_vars(preds).values())
                loss.backward()
        torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t0)
        if train:
            model.zero_grad(set_to_none=True)
    model.eval()
    times.sort()
    return times[len(times) // 2], torch.cuda.max_memory_allocated() / 2**30


def main():
    torch.manual_seed(0)
    surf_vars, atmos_vars, static_vars = inference.gather_vars_for_model(args.dataset_config_path, args.data_sources)
    model = build_model(surf_vars, atmos_vars, static_vars).eval()
    loader = inference.load_data(args.dataset_config_path, args.test_data_sources)

    worst_one_step, worst_rollout = 0.0, [0.0] * own_args.rollout_steps
    first_batch = None
    for i, (x, y) in enumerate(loader):
        if i >= own_args.n_samples:
            break
        x = drop_unknown_vars(x, surf_vars, atmos_vars, static_vars)
        if first_batch is None:
            first_batch = x
        new = predict(model, x)
        with old_path():
            old = predict(model, x)
            old2 = predict(model, x)
        target = all_vars(y)
        print(f"\n=== sample {i} ({x.metadata.time[-1]}) ===")
        print(f"{'variable':<16}{'rel new-old':>14}{'rel old-old':>14}{'rmse old':>14}{'rmse new':>14}")
        for k in new:
            d = rel_diff(new[k], old[k])
            worst_one_step = max(worst_one_step, d)
            tk = target.get(k)
            r_old = rmse(old[k], tk) if tk is not None else float("nan")
            r_new = rmse(new[k], tk) if tk is not None else float("nan")
            print(f"{k:<16}{d:>14.2e}{rel_diff(old2[k], old[k]):>14.2e}{r_old:>14.5g}{r_new:>14.5g}")

        if own_args.rollout_steps > 0:
            ro_new = predict_rollout(model, x, own_args.rollout_steps)
            with old_path():
                ro_old = predict_rollout(model, x, own_args.rollout_steps)
            print("rollout step: max / median rel diff new-old over variables")
            for s, (a, b) in enumerate(zip(ro_new, ro_old)):
                ds = sorted(rel_diff(a[k], b[k]) for k in a)
                worst_rollout[s] = max(worst_rollout[s], ds[-1])
                print(f"  step {s + 1:>2}: {ds[-1]:.2e} / {ds[len(ds) // 2]:.2e}")

    print(f"\nworst one-step rel diff over samples and variables: {worst_one_step:.2e}")
    if own_args.rollout_steps > 0:
        print("worst rollout rel diff per step: " + ", ".join(f"{d:.1e}" for d in worst_rollout))

    print("\n=== benchmark (median step time, peak memory) ===")
    modes = [("inference", False)] + ([] if own_args.skip_train_bench else [("train fwd+bwd", True)])
    for name, train in modes:
        if train:
            model.configure_activation_checkpointing()  # as train.py does for the backbone
        with old_path():
            t_old, m_old = time_step(model, first_batch, train)
        t_new, m_new = time_step(model, first_batch, train)
        print(f"{name:<14} old {t_old * 1e3:8.1f} ms {m_old:6.1f} GiB | new {t_new * 1e3:8.1f} ms {m_new:6.1f} GiB "
              f"| {100 * (t_new / t_old - 1):+.1f}% time, {m_new - m_old:+.2f} GiB")


if __name__ == "__main__":
    main()
