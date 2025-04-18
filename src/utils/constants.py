class Constants:
    """
    Constants for benchmark types and evaluation modes.
    """

    # Models
    codegen_tokenizer = "Salesforce/codegen-6B-mono"
    codex_tokenizer = "p50k_base"

    # Regular benchmark identifiers for Codex
    api_benchmark: str = "random_api"
    line_benchmark: str = "random_line"

    # Shorter variants for CodeGen
    short_api_benchmark: str = "short_api"
    short_line_benchmark: str = "short_line"

    # Evaluation modes
    gt: str = "gt"  # Ground truth
    rg: str = "r-g"  # Retrieve-and-generate
    rgrg: str = "r-g-r-g"  # Two-stage RAG (RepoCoder)

    # Base level directories
    base_repos_dir: str = "data/repositories"
    base_datasets_dir: str = "data/datasets"
    base_cache_windows_dir: str = "data/cache/window"
    base_predictions_dir = "data/predictions"

    # TODO: fix this path
    repo_base_dir: str = "data/repositories/line_and_api_level"

    # Default benchmark task file paths
    api_completion_benchmark: str = (
        f"{base_datasets_dir}/api_level_completion_2k_context_codex.test.jsonl"
    )
    random_line_completion_benchmark: str = (
        f"{base_datasets_dir}/line_level_completion_2k_context_codex.test.jsonl"
    )

    short_api_completion_benchmark: str = (
        f"{base_datasets_dir}/api_level_completion_1k_context_codegen.test.jsonl"
    )
    short_random_line_completion_benchmark: str = (
        f"{base_datasets_dir}/line_level_completion_1k_context_codegen.test.jsonl"
    )
