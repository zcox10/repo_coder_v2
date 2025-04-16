from typing import List, Dict, Any
from collections import defaultdict

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class VectorUtils:
    @staticmethod
    def resolve_repo_window_paths(
        repos: List[str], window_sizes: List[int], slice_size: int
    ) -> List[str]:
        """
        Returns a list of file paths for repository-level window pickle files.
        """
        paths = []
        for window_size in window_sizes:
            for repo in repos:
                path = FilePathBuilder.repo_windows_path(repo, window_size, slice_size)
                paths.append(path)
        return paths

    @staticmethod
    def resolve_search_window_paths(
        repos: List[str], window_sizes: List[int], slice_sizes: List[int], benchmark: str, mode: str
    ) -> List[str]:
        """
        Returns a list of file paths for task-based (RG1/GT) search window pickle files.
        """
        paths = []
        for window_size in window_sizes:
            for slice_size in slice_sizes:
                for repo in repos:
                    path = FilePathBuilder.search_first_window_path(
                        benchmark, mode, repo, window_size, slice_size
                    )
                    paths.append(path)
        return paths

    @staticmethod
    def resolve_prediction_window_paths(
        repos: List[str],
        window_sizes: List[int],
        slice_size: int,
        benchmark: str,
        mode: str,
        prediction_path: str,
    ) -> List[str]:
        """
        Returns a list of file paths for prediction-based (RepoCoder) window pickle files.
        """
        paths = []
        for window_size in window_sizes:
            for repo in repos:
                path = FilePathBuilder.gen_first_window_path(
                    benchmark, mode, prediction_path, repo, window_size, slice_size
                )
                paths.append(path)
        return paths

    @staticmethod
    def get_input_lines_from_window_file(window_file_path: str) -> List[Dict[str, Any]]:
        """
        Loads window lines from a pickle file and transforms them into embedding input format.
        """
        lines = Tools.load_pickle(window_file_path)
        return [
            {
                "context": line["context"],
                "metadata": {
                    "window_file_path": window_file_path,
                    "original_metadata": line["metadata"],
                },
            }
            for line in lines
        ]

    @staticmethod
    def get_input_lines_for_window_files(window_files: List[str]) -> List[Dict[str, Any]]:
        """
        Collects and flattens embedding input lines from a list of window file paths.
        """
        all_lines = []
        for window_file in window_files:
            all_lines.extend(VectorUtils.get_input_lines_from_window_file(window_file))
        return all_lines

    @staticmethod
    def get_input_lines_for_repo_windows(
        repos: List[str], window_sizes: List[int], slice_size: int
    ) -> List[Dict[str, Any]]:
        paths = VectorUtils.resolve_repo_window_paths(repos, window_sizes, slice_size)
        return VectorUtils.get_input_lines_for_window_files(paths)

    @staticmethod
    def get_input_lines_for_baseline_and_ground(
        repos: List[str], window_sizes: List[int], slice_sizes: List[int], benchmark: str, mode: str
    ) -> List[Dict[str, Any]]:
        paths = VectorUtils.resolve_search_window_paths(
            repos, window_sizes, slice_sizes, benchmark, mode
        )
        return VectorUtils.get_input_lines_for_window_files(paths)

    @staticmethod
    def get_input_lines_for_predictions(
        repos: List[str],
        window_sizes: List[int],
        slice_size: int,
        benchmark: str,
        mode: str,
        prediction_path: str,
    ) -> List[Dict[str, Any]]:
        paths = VectorUtils.resolve_prediction_window_paths(
            repos, window_sizes, slice_size, benchmark, mode, prediction_path
        )
        return VectorUtils.get_input_lines_for_window_files(paths)

    @staticmethod
    def place_generated_embeddings(generated_embeddings: List[Dict[str, Any]]) -> None:
        """
        Groups embeddings by output path and writes them to disk using ada002 path conventions.
        """
        vector_file_path_to_lines = defaultdict(list)
        for line in generated_embeddings:
            window_path = line["metadata"]["window_file_path"]
            original_metadata = line["metadata"]["original_metadata"]
            vector_file_path = FilePathBuilder.ada002_vector_path(window_path)
            vector_file_path_to_lines[vector_file_path].append(
                {
                    "context": line["context"],
                    "metadata": original_metadata,
                    "data": line["data"],
                }
            )

        for vector_file_path, lines in vector_file_path_to_lines.items():
            Tools.dump_pickle(lines, vector_file_path)
