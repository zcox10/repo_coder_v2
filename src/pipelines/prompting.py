from typing import List

from src.build_prompts.build_prompt_wrapper import BuildPromptWrapper
from src.utils.constants import Constants
from src.utils.codegen_tokenizer import CodeGenTokenizer


def build_prompts_for_baseline_and_ground(
    benchmark: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    vector_type: str = "one-gram",
    tokenizer_cls=CodeGenTokenizer,
) -> None:
    """
    Builds prompts for inference based on baseline (RG1) and ground-truth (GT) retrieval results.

    Args:
        benchmark: Benchmark identifier (e.g., "short_api_benchmark").
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of stride values.
        vector_type: Vector type used for retrieval (e.g., 'one-gram').
        tokenizer_cls: Tokenizer class to use (default: CodeGenTokenizer).
    """
    for window_size in window_sizes:
        for slice_size in slice_sizes:
            for mode in [Constants.rg, Constants.gt]:
                output_file_path = (
                    f"data/prompts/{mode}-{vector_type}-ws-{window_size}-ss-{slice_size}.jsonl"
                )
                BuildPromptWrapper(
                    vector_type, benchmark, repos, window_size, slice_size, tokenizer_cls
                ).build_first_search_prompt(mode, output_file_path)


def build_prompts_for_predictions(
    benchmark: str,
    repos: List[str],
    window_sizes: List[int],
    slice_sizes: List[int],
    mode: str,
    prediction_path_template: str,
    vector_type: str = "one-gram",
    tokenizer_cls=CodeGenTokenizer,
) -> None:
    """
    Builds prompts for inference using windows generated from predicted completions (e.g., RepoCoder).

    Args:
        benchmark: Benchmark identifier.
        repos: List of repository names.
        window_sizes: List of context window sizes.
        slice_sizes: List of stride values.
        mode: Evaluation mode (e.g., 'r-g-r-g').
        prediction_path_template: Format string for prediction JSONL file.
        vector_type: Vector type used for retrieval (e.g., 'one-gram').
        tokenizer_cls: Tokenizer class to use (default: CodeGenTokenizer).
    """
    for window_size in window_sizes:
        for slice_size in slice_sizes:
            prediction_path = prediction_path_template.format(
                window_size=window_size, slice_size=slice_size
            )
            output_file_path = (
                f"data/prompts/repocoder-{vector_type}-ws-{window_size}-ss-{slice_size}.jsonl"
            )
            BuildPromptWrapper(
                vector_type, benchmark, repos, window_size, slice_size, tokenizer_cls
            ).build_prediction_prompt(mode, prediction_path, output_file_path)
