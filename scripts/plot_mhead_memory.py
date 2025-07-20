#!/usr/bin/env python3
"""Simple memory usage comparison for CP tensor functions."""

import os
from typing import Literal, Callable
import time
import psutil
from tqdm import tqdm

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mtp.mheads import MHEADS
from mtp.mheads._abc import AbstractDisributionHeadConfig

sns.set_theme()


def get_peak_memory_usage(fn: Callable, device: str = "cuda", **kwargs) -> float:
    """Get the peak memory usage of a function.

    Args:
        fn (Callable): The function to measure the memory usage of.
        device (str): The device to measure the memory usage on.
        kwargs (dict): The keyword arguments to pass to the function.

    Returns:
        float: The peak memory usage of the function in MB.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fn(**kwargs)

    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
    else:
        memory_mb = psutil.Process().memory_info().rss / (1024**2)

    return memory_mb


def get_latency(fn: Callable, device: str = "cuda", **kwargs) -> float:
    """Get the latency of a function.

    Args:
        fn (Callable): The function to measure the latency of.
        device (str, optional): The device to measure the latency on. Defaults to "cuda".
        kwargs (dict): The keyword arguments to pass to the function.

    Returns:
        float: The latency of the function in seconds.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(3):
        fn(**kwargs)

    start = time.perf_counter()
    fn(**kwargs)
    # wait for cuda to finish if needed
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start  # in seconds


def train_fn(
    batch_size: int,
    horizon: int,
    rank: int,
    model_head: str = "stp",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    embedding_dim=256,
    vocab_size=30000,
    mode: Literal["init", "forward", "backward"] = "init",
    **kwargs,
):
    config = AbstractDisributionHeadConfig(
        d_model=embedding_dim,
        d_output=vocab_size,
        horizon=horizon,
        rank=rank,
    )
    model = MHEADS[model_head](config).to(device)
    x = torch.randn(batch_size, embedding_dim, device=device)
    y = torch.randint(0, vocab_size, (batch_size, horizon), device=device)

    if mode in ["forward", "backward"]:
        output = model(x, y)
        if mode in ["backward"]:
            loss = output.loss.mean()
            loss.backward()


def main():

    # Default values
    defaults = {
        "batch_size": 128,
        "horizon": 2,
        "rank": 2,
        "embedding_dim": 768,
        "vocab_size": 30000,
    }
    max_exps = {
        "horizon": 5,
        "rank": 5,
        # "embedding_dim": 2,
        "batch_size": 5,
    }

    # Attrs:
    # col: hparam (batch_size, horizon, rank, embedding_dim)
    # x: multiplier
    # y: memory_mb
    # hue: mode (init, init + forward, init + forward + backward)
    # style: model_head (cp, cp_drop)

    kwargs = []
    for mode in ["init", "forward", "backward"]:
        for head in ["multihead"]:
            for hparam, max_exp in max_exps.items():
                for i in range(max_exp):
                    kwargs.append(
                        {
                            **defaults,
                            "hparam": hparam,
                            "multiplier": 2**i,
                            "mode": mode,
                            "model_head": head,
                            hparam: 2**i * defaults[hparam],
                        }
                    )

    # Add STP configs
    for i in range(max_exps["batch_size"]):
        kwargs.append(
            {
                **defaults,
                "hparam": "batch_size",
                "multiplier": 2**i,
                "model_head": "stp",
                "horizon": 1,
                "rank": 1,
            }
        )

    # Compute memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pbar = tqdm(kwargs, desc="Running forward pass")
    for conf in pbar:
        memory_mb = get_peak_memory_usage(train_fn, device=device, **conf)
        latency_sec = get_latency(train_fn, device=device, **conf)
        conf["memory_mb"] = memory_mb
        conf["latency_sec"] = latency_sec
        pbar.set_postfix({"model_head": conf["model_head"]})

    df = pd.DataFrame(kwargs)
    sns.relplot(
        data=df,
        x="multiplier",
        y="memory_mb",
        col="hparam",
        style="model_head",
        hue="mode",
        kind="line",
        markers=True,
        alpha=0.6,
    )
    plt.xscale("log", base=2)
    plt.savefig("results/plots/mhead_memory_usage_facetgrid.png")
    print(f"Plot saved to results/plots/mhead_memory_usage_facetgrid.png")

    # Plot latency
    sns.relplot(
        data=df,
        x="multiplier",
        y="latency_sec",
        col="hparam",
        style="model_head",
        hue="mode",
        kind="line",
        markers=True,
        alpha=0.6,
    )
    plt.xscale("log", base=2)
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/mhead_latency_facetgrid.png")
    print(f"Plot saved to results/plots/mhead_latency_facetgrid.png")


if __name__ == "__main__":
    main()
