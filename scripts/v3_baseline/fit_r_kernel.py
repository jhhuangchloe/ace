"""Fit the temporal noise kernel of the two-block ``r`` block to measured residuals.

Reads the covariances saved by ``r_statistics.py`` and compares, per channel, the
interior (unit-diagonal) correlation matrix of ``r`` with the kernels the code can build
for the endpoint-pinned block: Brownian bridge, independent noise, and OU with a chosen
length (in units of the clip duration). Prints the best OU length per channel (config
field ``r_kernel_length_scale_by_channel``) and the relative misfit (0 = perfect).

    python fit_r_kernel.py <cov.npz>
"""

import sys

import numpy as np

N = 7  # interior steps of a 9-step clip
TIMES = np.arange(1, N + 1) / (N + 1)
LENGTHS = np.round(np.arange(0.1, 0.81, 0.025), 3)


def correlation(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def brownian_bridge() -> np.ndarray:
    return correlation(np.minimum.outer(TIMES, TIMES) - np.outer(TIMES, TIMES))


def ou(length: float) -> np.ndarray:
    return np.exp(-np.abs(TIMES[:, None] - TIMES[None, :]) / length)


def misfit(target: np.ndarray, kernel: np.ndarray) -> float:
    return float(np.linalg.norm(target - kernel) / np.linalg.norm(target))


def main(path: str, baseline: str = "base") -> None:
    data = np.load(path)
    channels = sorted({k.split("|")[2] for k in data.files})
    corr = {
        c: correlation(
            (data[f"train|{baseline}|{c}"] + data[f"val|{baseline}|{c}"]) / 2
        )
        for c in channels
    }
    bridge, independent = brownian_bridge(), np.eye(N)
    best = {}
    print(f"{'channel':34}{'bridge':>8}{'indep.':>8}{'best OU':>9}{'length':>8}")
    for c in channels:
        errors = [misfit(corr[c], ou(length)) for length in LENGTHS]
        j = int(np.argmin(errors))
        best[c] = float(LENGTHS[j])
        print(
            f"{c:34}{misfit(corr[c], bridge):8.3f}{misfit(corr[c], independent):8.3f}"
            f"{errors[j]:9.3f}{LENGTHS[j]:8.3f}"
        )
    shared = [
        np.mean([misfit(corr[c], ou(length)) for c in channels]) for length in LENGTHS
    ]
    j = int(np.argmin(shared))
    print(
        "\nmean misfit: bridge "
        f"{np.mean([misfit(corr[c], bridge) for c in channels]):.3f}, per-channel OU "
        f"{np.mean([misfit(corr[c], ou(best[c])) for c in channels]):.3f}, "
        f"one shared OU (length {LENGTHS[j]}) {shared[j]:.3f}"
    )
    print("\nconfig snippet:\n  r_kernel: ou\n  r_kernel_length_scale_by_channel:")
    for c in channels:
        print(f"    {c}: {best[c]}")


if __name__ == "__main__":
    main(sys.argv[1])
