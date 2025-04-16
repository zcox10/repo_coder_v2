# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
from typing import Callable, List

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder


class BuildVectorWrapper:
    """
    Coordinates vectorization of context windows across multiple repos, benchmarks,
    and window/slice configurations using a given vector builder (e.g., BagOfWords).
    """

    def __init__(
        self,
        benchmark: str,
        vector_builder: Callable[[str], object],
        repos: List[str],
        window_sizes: List[int],
        slice_sizes: List[int],
    ):
        """
        Args:
            benchmark: Benchmark name (e.g., short_api_benchmark)
            vector_builder: A callable class or function that implements `.build()`
            repos: List of repository names
            window_sizes: List of context window sizes
            slice_sizes: List of stride sizes
        """
        self.benchmark = benchmark
        self.vector_builder = vector_builder
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes

    def vectorize_repo_windows(self) -> None:
        """
        Vectorizes context windows created directly from repository source code.
        """
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                builder = self.vector_builder(path)
                builder.build()

    def vectorize_baseline_and_ground_windows(self) -> None:
        """
        Vectorizes context windows for both baseline (RG1) and ground-truth (GT) modes.
        """
        for slice_size in self.slice_sizes:
            for window_size in self.window_sizes:
                for repo in self.repos:
                    # RG1 mode
                    rg_path = FilePathBuilder.search_first_window_path(
                        self.benchmark, Constants.rg, repo, window_size, slice_size
                    )
                    self.vector_builder(rg_path).build()

                    # GT mode
                    gt_path = FilePathBuilder.search_first_window_path(
                        self.benchmark, Constants.gt, repo, window_size, slice_size
                    )
                    self.vector_builder(gt_path).build()

    def vectorize_prediction_windows(self, mode: str, prediction_path_template: str) -> None:
        """
        Vectorizes context windows that are derived from model predictions (e.g., RepoCoder).

        Args:
            mode: Evaluation mode (typically 'r-g-r-g')
            prediction_path_template: Template string to format the path to prediction JSONL
        """
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(
                window_size=window_size, slice_size=slice_size
            )
            for repo in self.repos:
                window_path = FilePathBuilder.gen_first_window_path(
                    self.benchmark, mode, prediction_path, repo, window_size
                )
                self.vector_builder(window_path).build()
