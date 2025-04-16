from typing import List

from src.build_retrievals.code_search_wrapper import CodeSearchWrapper


def search_baseline_and_ground(
    benchmark: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    vector_type: str = "one-gram",
) -> None:
    """
    Performs vector-based retrieval for both baseline (RG1) and ground truth (GT) modes.

    Args:
        benchmark: Benchmark identifier (e.g., "short_api_benchmark").
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of slicing strides.
        vector_type: Embedding type used for retrieval (default: 'one-gram').
    """
    CodeSearchWrapper(
        vector_type, benchmark, repos, window_sizes, slice_sizes
    ).search_baseline_and_ground()


def search_predictions(
    benchmark: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    mode: str,
    prediction_path_template: str,
    vector_type: str = "one-gram",
) -> None:
    """
    Performs vector-based retrieval for prediction-derived windows (e.g., RepoCoder).

    Args:
        benchmark: Benchmark identifier.
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of slicing strides.
        mode: Evaluation mode (e.g., 'r-g-r-g').
        prediction_path_template: Template string for prediction path.
        vector_type: Embedding type used for retrieval (default: 'one-gram').
    """
    CodeSearchWrapper(vector_type, benchmark, repos, window_sizes, slice_sizes).search_prediction(
        mode, prediction_path_template
    )
