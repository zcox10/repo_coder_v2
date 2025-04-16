"""
Windowing stage of the pipeline.

This module provides methods for generating:
- Repository-based context windows
- Baseline and ground-truth windows
- Prediction-derived context windows
"""

from typing import List
from src.build_windows.make_window import MakeWindowWrapper


def make_repo_windows(
    base_dir: str, repos: List[str], window_sizes: List[int], slice_sizes: List[int]
) -> None:
    """
    Builds context windows directly from repository source files.

    Args:
        base_dir: Base directory where the repositories are stored.
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of stride values for slicing.
    """
    MakeWindowWrapper(None, base_dir, repos, window_sizes, slice_sizes).window_for_repo_files()


def make_baseline_and_ground_windows(
    benchmark: str, base_dir: str, repos: List[str], window_sizes: List[int], slice_sizes: List[int]
) -> None:
    """
    Builds task-based windows for both the baseline (RG1) and oracle (GT) methods.

    Args:
        benchmark: Benchmark identifier (e.g., "short_api_benchmark").
        base_dir: Base directory where the repositories are stored.
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of stride values.
    """
    MakeWindowWrapper(
        benchmark, base_dir, repos, window_sizes, slice_sizes
    ).window_for_baseline_and_ground()


def make_prediction_windows(
    benchmark: str,
    base_dir: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    mode: str,
    prediction_path_template: str,
) -> None:
    """
    Builds windows from predicted completions (e.g., RepoCoder outputs).

    Args:
        benchmark: Benchmark identifier.
        base_dir: Base directory where the repositories are stored.
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of stride values.
        mode: Evaluation mode (e.g., 'r-g-r-g').
        prediction_path_template: Format string for prediction path (e.g., 'predictions/...-ws-{window_size}-ss-{slice_size}.jsonl').
    """
    MakeWindowWrapper(benchmark, base_dir, repos, window_sizes, slice_sizes).window_for_prediction(
        mode, prediction_path_template
    )
