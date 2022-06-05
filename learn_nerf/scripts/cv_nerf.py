"""
Run a NeRF model with K-fold cross-validation to find the frames which have
the highest validation loss. These might be samples in the dataset with
incorrect camera poses.
"""

import argparse
import random
import sys
import tempfile
from typing import Iterator, List, Set

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray as KeyArray
from learn_nerf.dataset import NeRFDataset, load_dataset
from learn_nerf.scripts.train_nerf import add_model_args, create_model
from learn_nerf.train import TrainLoop
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096, help="rays per batch")
    parser.add_argument(
        "--folds", type=int, default=10, help="number of training runs to perform"
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
    parser.add_argument("--train_iters", type=int, default=1500)
    add_model_args(parser)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    print("loading dataset...")
    data = load_dataset(args.data_dir)

    global_key = jax.random.PRNGKey(
        args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
    )
    init_key, shuffle_key, global_key = jax.random.split(global_key, num=3)
    shuffle_indices = jax.random.permutation(
        shuffle_key, jnp.arange(len(data.views))
    ).tolist()

    for i, valid_indices in enumerate(chunk_indices(args.folds, shuffle_indices)):
        print(f"performing cross validation for fold {i}...")
        train_data = NeRFDataset(
            metadata=data.metadata,
            views=[x for i, x in enumerate(data.views) if i not in valid_indices],
        )
        valid_data = NeRFDataset(
            metadata=data.metadata,
            views=[x for i, x in enumerate(data.views) if i in valid_indices],
        )
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
        step_fn = loop.step_fn(
            jnp.array(data.metadata.bbox_min),
            jnp.array(data.metadata.bbox_max),
        )
        key = global_key
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_key, key = jax.random.split(key, 2)
            batch_iter = train_data.iterate_batches(tmp_dir, data_key, args.batch_size)
            batch = next(batch_iter)
            print("dataset shuffling complete.")
            for _ in tqdm(range(args.train_iters), file=sys.stderr):
                step_key, key = jax.random.split(key, 2)
                step_fn(step_key, batch)
                batch = next(batch_iter)
        valid_results = validation_losses(
            key=key, loop=loop, data=valid_data, batch_size=args.batch_size
        )
        for view, loss in zip(valid_data.views, valid_results):
            print(loss, view.image_path)


def validation_losses(
    key: KeyArray, loop: TrainLoop, data: NeRFDataset, batch_size: int
) -> Iterator[float]:
    loss_fn = jax.jit(
        lambda key, batch, params: loop.losses(
            key=key,
            bbox_min=jnp.array(data.metadata.bbox_min),
            bbox_max=jnp.array(data.metadata.bbox_max),
            batch=batch,
            params=params,
        )[1]
    )
    for view in data.views:
        rays = view.rays()
        total_loss = 0.0
        for i in range(0, rays.shape[0], batch_size):
            test_key, key = jax.random.split(key)
            sub_batch = rays[i : i + batch_size]
            losses = loss_fn(test_key, sub_batch, loop.state.params)
            total_loss += float(losses["fine"]) * len(sub_batch)
        yield total_loss / rays.shape[0]


def chunk_indices(num_chunks: int, indices: List[int]) -> Iterator[Set[int]]:
    chunk_size = len(indices) // num_chunks
    extra = len(indices) % num_chunks
    offset = 0
    for i in range(num_chunks):
        if i < extra:
            size = chunk_size + 1
        else:
            size = chunk_size
        if not size:
            return
        yield set(indices[offset : offset + size])
        offset += size
    assert offset == len(indices)


if __name__ == "__main__":
    main()
