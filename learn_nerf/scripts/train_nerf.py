"""
Train a NeRF model on a scene.
"""

import argparse
import os
import random
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from learn_nerf.dataset import ModelMetadata, load_dataset
from learn_nerf.instant_ngp import InstantNGPModel, InstantNGPRefNERFModel
from learn_nerf.model import ModelBase, NeRFModel
from learn_nerf.ref_nerf import RefNERFModel
from learn_nerf.train import TrainLoop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096, help="rays per batch")
    parser.add_argument(
        "--test_batch_size", type=int, default=None, help="rays per test batch"
    )
    parser.add_argument(
        "--coarse_samples", type=int, default=64, help="samples per coarse ray"
    )
    parser.add_argument(
        "--fine_samples",
        type=int,
        default=128,
        help="samples per fine ray (not including coarse samples)",
    )
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="nerf.pkl")
    parser.add_argument("--one_view", action="store_true")
    parser.add_argument("--test_data_dir", type=str, default=None)
    add_model_args(parser)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    print("loading dataset...")
    data = load_dataset(args.data_dir)
    if args.one_view:
        data.views = data.views[:1]

    if args.test_data_dir is not None:
        print("loading test dataset...")
        test_data = load_dataset(args.test_data_dir)
        if args.one_view:
            test_data.views = test_data.views[:1]
    else:
        test_data = None

    key = jax.random.PRNGKey(
        args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
    )
    init_key, key = jax.random.split(key)

    print("creating model and train loop...")
    coarse, fine, train_kwargs = create_model(args, data.metadata)
    loop = TrainLoop(
        coarse,
        fine,
        init_rng=init_key,
        lr=args.lr,
        coarse_ts=args.coarse_samples,
        fine_ts=args.fine_samples,
        **train_kwargs,
    )
    if os.path.exists(args.save_path):
        print(f"loading from checkpoint: {args.save_path}")
        loop.load(args.save_path)
    step_fn = loop.step_fn(
        jnp.array(data.metadata.bbox_min),
        jnp.array(data.metadata.bbox_max),
    )
    if test_data is not None:
        loss_fn = jax.jit(
            lambda key, batch, params: loop.losses(
                key=key,
                bbox_min=jnp.array(data.metadata.bbox_min),
                bbox_max=jnp.array(data.metadata.bbox_max),
                batch=batch,
                params=params,
            )[1]
        )

    print("training...")
    data_key, test_data_key, key = jax.random.split(key, 3)
    shuffle_dir = os.path.join(args.data_dir, "shuffled")
    if test_data:
        test_shuffle_dir = os.path.join(args.test_data_dir, "shuffled")
        test_iterator = test_data.iterate_batches(
            test_shuffle_dir, test_data_key, args.test_batch_size
        )
    for i, batch in enumerate(
        data.iterate_batches(shuffle_dir, data_key, args.batch_size)
    ):
        step_key, test_key, key = jax.random.split(key, 3)
        if test_data is not None:
            test_batch = next(test_iterator)
            test_losses = {
                f"test_{k}": v
                for k, v in loss_fn(test_key, test_batch, loop.state.params).items()
            }
        losses = step_fn(step_key, batch)
        if test_data is not None:
            losses.update(test_losses)
        loss_str = " ".join(f"{k}={float(v):.05}" for k, v in losses.items())
        print(f"step {i}: {loss_str}")
        if i and i % args.save_interval == 0:
            loop.save(args.save_path)


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--instant_ngp", action="store_true")
    parser.add_argument("--ref_nerf", action="store_true")


def create_model(
    args: argparse.Namespace, metadata: ModelMetadata
) -> Tuple[ModelBase, ModelBase, Dict[str, Any]]:
    if args.instant_ngp:
        if args.ref_nerf:
            model_cls = partial(InstantNGPRefNERFModel, sh_degree=4)
        else:
            model_cls = InstantNGPModel
        coarse = model_cls(
            table_sizes=[2 ** 18] * 6,
            grid_sizes=[2 ** (4 + i // 2) for i in range(6)],
            bbox_min=jnp.array(metadata.bbox_min),
            bbox_max=jnp.array(metadata.bbox_max),
        )
        fine = model_cls(
            table_sizes=[2 ** 18] * 16,
            grid_sizes=[2 ** (4 + i // 2) for i in range(16)],
            bbox_min=jnp.array(metadata.bbox_min),
            bbox_max=jnp.array(metadata.bbox_max),
        )
        train_kwargs = dict(adam_eps=1e-15, adam_b1=0.9, adam_b2=0.99)
    else:
        if args.ref_nerf:
            model_cls = partial(RefNERFModel, sh_degree=4)
        else:
            model_cls = NeRFModel
        coarse = model_cls()
        fine = model_cls()
        train_kwargs = dict()
    return coarse, fine, train_kwargs


if __name__ == "__main__":
    main()
