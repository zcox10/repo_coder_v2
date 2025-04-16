from typing import List

from src.build_vectors.bag_of_words import BagOfWords
from src.build_vectors.build_vector import BuildVectorWrapper


def vectorize_repo_windows(
    repos: List[str], window_sizes: List[int], slice_sizes: List[int]
) -> None:
    """
    Vectorizes windows generated from raw repository files.

    Args:
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of slicing strides.
    """
    vectorizer = BagOfWords
    BuildVectorWrapper(None, vectorizer, repos, window_sizes, slice_sizes).vectorize_repo_windows()


def vectorize_baseline_and_ground_windows(
    benchmark: str, repos: List[str], window_sizes: List[int], slice_sizes: List[int]
) -> None:
    """
    Vectorizes windows for both baseline (RG1) and ground truth (GT) modes.

    Args:
        benchmark: Benchmark name (e.g., "short_api_benchmark").
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of slicing strides.
    """
    vectorizer = BagOfWords
    BuildVectorWrapper(
        benchmark, vectorizer, repos, window_sizes, slice_sizes
    ).vectorize_baseline_and_ground_windows()


def vectorize_prediction_windows(
    benchmark: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    mode: str,
    prediction_path_template: str,
) -> None:
    """
    Vectorizes windows generated from model predictions (e.g., RepoCoder).

    Args:
        benchmark: Benchmark name.
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of slicing strides.
        mode: Evaluation mode (e.g., 'r-g-r-g').
        prediction_path_template: Format string for prediction path.
    """
    vectorizer = BagOfWords
    BuildVectorWrapper(
        benchmark, vectorizer, repos, window_sizes, slice_sizes
    ).vectorize_prediction_windows(mode, prediction_path_template)
