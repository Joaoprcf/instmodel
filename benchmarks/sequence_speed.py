"""Training throughput of a sequence-model backend on synthetic rows, in sequences per second."""

import argparse
import json
import time

import numpy as np

from instmodel import sequence


def synthetic_rows(args):
    rng = np.random.default_rng(0)
    start = rng.integers(0, args.length // 2, (args.rows, 1))
    groups = np.where(np.arange(args.length) < start, -1, rng.integers(0, 3, (args.rows, args.length)))
    return dict(tokens=rng.integers(0, args.vocab, (args.rows, args.length + 1)), groups=groups,
                weight=rng.random(args.rows))


def torch_backend(args, config):
    import torch

    from instmodel.sequence import torch as backend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {name: torch.as_tensor(value, device=device) for name, value in synthetic_rows(args).items()}
    torch.manual_seed(0)
    model = backend.Transformer(config, members=args.members).to(device)
    trainer = backend.Trainer(model, rows, args.batch, 1e-3, compile=args.compile)

    def run(steps):
        for _ in range(steps):
            trainer.step(1e-3)
        if device == "cuda":
            torch.cuda.synchronize()

    return run, device


def tf_backend(args, config):
    import tensorflow as tf

    from instmodel.sequence import tf as backend

    device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
    model = backend.Transformer(config, members=args.members)
    trainer = backend.Trainer(model, synthetic_rows(args), args.batch, 1e-3, compile=not args.no_xla)

    def run(steps):
        for _ in range(steps):
            trainer.step(1e-3)
        trainer.loss.numpy()  # Waits for the queued steps.

    return run, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "tf"), default="torch")
    parser.add_argument("--vocab", type=int, default=28)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--ff-width", type=int, default=0, help="default: the width")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feature-rows", type=int, default=24, help="0 disables segment features")
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64, help="rows per member and step")
    parser.add_argument("--length", type=int, default=112, help="supervised positions per row")
    parser.add_argument("--rows", type=int, default=200_000, help="synthetic rows sampled from")
    parser.add_argument("--seconds", type=float, default=20, help="timed steady-state duration")
    parser.add_argument("--compile", action="store_true", help="torch: Inductor-compile the loss (slow first build)")
    parser.add_argument("--no-xla", action="store_true", help="tf: run the step as a graph without XLA")
    args = parser.parse_args()
    features = None
    if args.feature_rows:
        features = sequence.SegmentFeatures(members=range(10), openers=(min(11, args.vocab - 1),),
                                            rows=args.feature_rows)
    config = sequence.TransformerConfig(vocab_size=args.vocab, width=args.width, heads=args.heads,
                                        kv_heads=args.kv_heads, ff_width=args.ff_width, layers=args.layers,
                                        features=features, max_positions=max(192, args.length))
    run, device = (tf_backend if args.backend == "tf" else torch_backend)(args, config)
    run(50)  # Warm caches and allocator before timing.
    steps, chunk, begin = 0, 20, time.perf_counter()
    while time.perf_counter() - begin < args.seconds:
        run(chunk)
        steps += chunk
    seconds = time.perf_counter() - begin
    print(json.dumps(dict(backend=args.backend, device=device, parameters=sequence.parameter_count(config),
                          steps_per_second=round(steps / seconds, 1),
                          sequences_per_second=round(steps * args.members * args.batch / seconds),
                          steps=steps, seconds=round(seconds, 2),
                          compile=not args.no_xla if args.backend == "tf" else args.compile)))


if __name__ == "__main__":
    main()
