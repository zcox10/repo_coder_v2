from typing import Any, Dict, List

from src.build_windows.base_window_maker import BaseWindowMaker

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class GroundTruthWindowMaker(BaseWindowMaker):
    """
    Constructs ground truth context windows for evaluation.

    For each task, it extracts a symmetric window of lines around the prediction line.
    This serves as the "ground truth" context used to compare against predicted windows.

    Args:
        benchmark (str): The benchmark name (e.g., "random_line").
        base_dir (str): The name of the base directory where repositories are stored.
        repo (str): The name of the repo the task belongs to.
        window_size (int): The number of total lines in the window.
        tasks (List[Dict[str, Any]]): List of task metadata dicts.
    """

    def __init__(
        self,
        benchmark: str,
        base_dir: str,
        repo: str,
        window_size: int,
        slice_size: int,
        tasks: List[Dict[str, Any]],
    ):
        super().__init__(base_dir, repo, window_size)
        self.benchmark = benchmark
        self.tasks = tasks
        self.slice_size = slice_size

    def build_window(self, print_lines: bool = False) -> None:
        """
        Builds and saves symmetric context windows centered around the task line.

        Args:
            print_lines (bool): If True, logs the start and end lines for each window.
        """
        code_windows: List[Dict[str, Any]] = []

        for task in self.tasks:
            # Skip tasks not belonging to this repo
            if task["metadata"]["task_id"].split("/")[0] != self.repo:
                continue

            fpath_tuple = tuple(task["metadata"]["fpath_tuple"])
            line_no: int = task["metadata"]["line_no"]
            context_start_lineno: int = task["metadata"]["context_start_lineno"]

            code_lines = self._get_code_lines(fpath_tuple)
            start_line, end_line = self._get_line_bounds(line_no, context_start_lineno)

            # Clamp to file length
            end_line = min(len(code_lines), end_line)

            context = self._get_context_window(code_lines, start_line, end_line)
            metadata = self._make_metadata(
                fpath_tuple=fpath_tuple,
                line_no=line_no,
                start_line=start_line,
                end_line=end_line,
                context_start_lineno=context_start_lineno,
                task_id=task["metadata"]["task_id"],
            )

            code_windows.append(
                {
                    "context": context,
                    "metadata": metadata,
                }
            )

            if print_lines:
                print("\nGROUND TRUTH:")
                print(f"START LINE: {start_line}")
                print(f"END LINE: {end_line}")

        self._log(
            f"Build {len(code_windows)} ground truth windows for {self.repo} with window size {self.window_size}"
        )

        output_path = FilePathBuilder.search_first_window_path(
            self.benchmark, Constants.gt, self.repo, self.window_size, self.slice_size
        )
        Tools.dump_pickle(code_windows, output_path)
