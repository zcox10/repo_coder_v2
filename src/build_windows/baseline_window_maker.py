from typing import Any, Dict, List

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.build_windows.base_window_maker import BaseWindowMaker


class BaselineWindowMaker(BaseWindowMaker):
    """
    Constructs context windows for the retrieve-and-generate baseline approach.

    For each task, extracts lines leading up to the target line (exclusive), forming a context
    window. These are later used for retrieval-augmented generation evaluation.

    Args:
        benchmark (str): The benchmark identifier (e.g., "random_line").
        base_dir (str): The name of the base directory where repositories are stored.
        repo (str): The name of the repository to process.
        window_size (int): Number of lines to include in the context window.
        tasks (List[Dict[str, Any]]): List of task dictionaries with metadata.
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
        Builds and saves baseline context windows for each matching task.
        The window includes all lines from (start_line) up to line_no (exclusive).

        Args:
            print_lines (bool): If True, prints window bounds for debugging.
        """
        code_windows: List[Dict[str, Any]] = []

        for task in self.tasks:
            # Skip tasks not from this repo
            if task["metadata"]["task_id"].split("/")[0] != self.repo:
                continue

            fpath_tuple = tuple(task["metadata"]["fpath_tuple"])
            line_no: int = task["metadata"]["line_no"]
            context_start_lineno: int = task["metadata"]["context_start_lineno"]

            code_lines = self._get_code_lines(fpath_tuple)
            start_line = max(context_start_lineno, line_no - self.window_size)
            end_line = line_no  # Exclude the line being predicted

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
                print("\nBASELINE:")
                print(f"START LINE: {start_line}")
                print(f"END LINE: {end_line}")

        self._log(
            f"Build {len(code_windows)} baseline windows for {self.repo} with window size {self.window_size}"
        )

        output_path = FilePathBuilder.search_first_window_path(
            self.benchmark, Constants.rg, self.repo, self.window_size, self.slice_size
        )
        Tools.dump_pickle(code_windows, output_path)
