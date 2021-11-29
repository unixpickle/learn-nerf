# learn-nerf

This is a JAX implementation of [Neural Radiance Fields](https://arxiv.org/abs/2003.08934) for learning purposes.

I've been curious about NeRF and its follow-up work for a while, but don't have much time to explore it. I learn best by doing, so I'll be implementing stuff here to try to get a feel for it.

# Usage

The steps to using this codebase are as follows:

 1. [Generate a dataset](#generating-a-dataset) - run a simple Go program to turn any `.stl` 3D model into a series of rendered camera views with associated metadata.
 2. [Train a model](#training-a-model) - install the Python dependencies and run the training script.

## Generating a dataset

I use a simple format for storing rendered views of the scene. Each frame is stored as a PNG file, and each PNG has an accompanying JSON file describing the camera view.

For easy experimentation, I create a Go program to render an arbitrary `.stl` file as a collection of views in the supported data format. To run this program, install [Go](https://go.dev/doc/install) and run `go get .` inside of [simple_dataset/](simple_dataset) to get the dependencies. Next, run

```
$ go run . /path/to/model.stl data_dir
```

This will create a directory `data_dir` containing rendered views of `/path/to/model.stl`.

## Training a model

First, install the `learn_nerf` package by running `pip install -e .` inside this repository. You should separately make sure [jax](https://github.com/google/jax) and [Flax](https://github.com/google/flax) are installed in your environment.

The training script is [learn_nerf/scripts/train_nerf.py](learn_nerf/scripts/train_nerf.py). Here's an example of running this script:

```
python learn_nerf/scripts/train_nerf \
    --lr 1e-5 \
    --batch_size 1024 \
    --save_path model_weights.pkl \
    /path/to/data_dir
```

This will periodically save model weights to `model_weights.pkl`. The script may get stuck on `training...` while it shuffles the dataset and compiles the training graph. Wait a minute or two, and losses should start printing out as training ramps up.

If you get a `Segmentation fault` on CPU, this may be because you don't have enough memory to run batch size 1024--try something lower.
