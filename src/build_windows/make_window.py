import itertools
import functools
from typing import List, Optional

from src.build_windows.baseline_window_maker import BaselineWindowMaker
from src.build_windows.ground_truth_window_maker import GroundTruthWindowMaker
from src.build_windows.prediction_window_maker import PredictionWindowMaker
from src.build_windows.repo_window_maker import RepoWindowMaker

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class MakeWindowWrapper:
    """
    High-level wrapper for generating context windows for different use cases:
    - Raw repository windows
    - Baseline (RG1) and ground-truth (GT) windows
    - Prediction-derived windows (e.g., RepoCoder)

    This class is used by `windowing.py` to drive window creation for the pipeline.
    """

    def __init__(
        self,
        benchmark: Optional[str],
        base_dir: str,
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ):
        """
        Initializes the wrapper with common configuration.

        Args:
            benchmark: Benchmark identifier or None (used only for task-based windows).
            repos: List of repositories to process.
            window_sizes: List of context window sizes.
            slice_sizes: List of slicing strides (for repo files or predictions).
        """
        self.benchmark = benchmark
        self.base_dir = base_dir
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes

        # Resolve path to the benchmark's task file
        if benchmark == Constants.line_benchmark:
            self.task_file_path = Constants.random_line_completion_benchmark
        elif benchmark == Constants.api_benchmark:
            self.task_file_path = Constants.api_completion_benchmark
        elif benchmark == Constants.short_line_benchmark:
            self.task_file_path = Constants.short_random_line_completion_benchmark
        elif benchmark == Constants.short_api_benchmark:
            self.task_file_path = Constants.short_api_completion_benchmark
        else:
            self.task_file_path = None  # Used only if benchmark is provided

    def window_for_repo_files(self) -> None:
        """
        Generates context windows from raw Python files in each repository.
        """
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                repo_window_maker = RepoWindowMaker(self.base_dir, repo, window_size, slice_size)
                repo_window_maker.build_windows()

    def window_for_baseline_and_ground(self) -> None:
        """
        Generates task-specific context windows for:
        - Baseline (RG1): truncated context
        - Ground truth (GT): full context up to target
        """
        if self.task_file_path is None:
            raise ValueError("No benchmark was provided to resolve task file path.")

        tasks = Tools.load_jsonl(self.task_file_path)

        for window_size in self.window_sizes:
            for slice_size in self.slice_sizes:
                for repo in self.repos:
                    BaselineWindowMaker(
                        self.benchmark, self.base_dir, repo, window_size, slice_size, tasks
                    ).build_window()

                    GroundTruthWindowMaker(
                        self.benchmark, self.base_dir, repo, window_size, slice_size, tasks
                    ).build_window()

    def window_for_prediction(self, mode: str, prediction_path_template: str) -> None:
        """
        Generates windows by inserting predictions at a target line (e.g., RepoCoder use case).

        Args:
            mode: Evaluation mode (e.g., 'r-g-r-g').
            prediction_path_template: Format string with {window_size} and {slice_size}.
        """
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(
                window_size=window_size, slice_size=slice_size
            )

            for repo in self.repos:
                window_path_builder = functools.partial(
                    FilePathBuilder.gen_first_window_path, self.benchmark, mode
                )
                pred_window_maker = PredictionWindowMaker(
                    self.base_dir, repo, window_size, prediction_path, window_path_builder
                )
                pred_window_maker.build_window()
