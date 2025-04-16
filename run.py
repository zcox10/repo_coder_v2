import os
from typing import List

# Disable parallel tokenization for HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.constants import Constants
from src.pipelines.windowing import (
    make_repo_windows,
    make_baseline_and_ground_windows,
    make_prediction_windows,
)
from src.pipelines.vectorization import (
    vectorize_repo_windows,
    vectorize_baseline_and_ground_windows,
    vectorize_prediction_windows,
)
from src.pipelines.retrieval import (
    search_baseline_and_ground,
    search_predictions,
)
from src.pipelines.prompting import (
    build_prompts_for_baseline_and_ground,
    build_prompts_for_predictions,
)


def run_repo_stage(
    base_dir: str, repos: List[str], window_sizes: List[int], slice_sizes: List[int]
) -> None:
    """
    Builds and vectorizes repo-level context windows.
    """
    make_repo_windows(base_dir, repos, window_sizes, slice_sizes)
    vectorize_repo_windows(repos, window_sizes, slice_sizes)


def run_rg1_and_gt_stage(
    benchmark: str,
    base_dir: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    vector_type: str = "one-gram",
) -> None:
    """
    Builds, vectorizes, retrieves, and constructs prompts for baseline and ground-truth windows.
    """
    make_baseline_and_ground_windows(benchmark, base_dir, repos, window_sizes, slice_sizes)
    vectorize_baseline_and_ground_windows(benchmark, repos, window_sizes, slice_sizes)
    search_baseline_and_ground(benchmark, repos, window_sizes, slice_sizes, vector_type)
    build_prompts_for_baseline_and_ground(benchmark, repos, window_sizes, slice_sizes, vector_type)


def run_repocoder_stage(
    benchmark: str,
    base_dir: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    prediction_path_template: str,
    mode: str = Constants.rgrg,
    vector_type: str = "one-gram",
) -> None:
    """
    Runs the full pipeline for the RepoCoder-style method (prediction-based generation).
    """
    make_prediction_windows(
        benchmark, base_dir, repos, window_sizes, slice_sizes, mode, prediction_path_template
    )
    vectorize_prediction_windows(
        benchmark, repos, window_sizes, slice_sizes, mode, prediction_path_template
    )
    search_predictions(
        benchmark, repos, window_sizes, slice_sizes, mode, prediction_path_template, vector_type
    )
    build_prompts_for_predictions(
        benchmark, repos, window_sizes, slice_sizes, mode, prediction_path_template, vector_type
    )


if __name__ == "__main__":
    # Repositories to evaluate
    repos = [
        "huggingface_diffusers",
        "nerfstudio-project_nerfstudio",
        # "awslabs_fortuna",
        # "huggingface_evaluate",
        # "google_vizier",
        # "alibaba_FederatedScope",
        # "pytorch_rl",
        # "opendilab_ACE",
    ]

    # Context window settings
    window_sizes = [20]
    slice_sizes = [2]  # Window stride = window_size / slice_size
    benchmark = Constants.short_api_benchmark
    prediction_path = (
        "data/predictions/rg-one-gram-ws-{window_size}-ss-{slice_size}_samples.0.jsonl"
    )

    run_repo_stage(Constants.base_repos_dir, repos, window_sizes, slice_sizes)
    run_rg1_and_gt_stage(benchmark, Constants.base_repos_dir, repos, window_sizes, slice_sizes)
    run_repocoder_stage(
        benchmark, Constants.base_repos_dir, repos, window_sizes, slice_sizes, prediction_path
    )
