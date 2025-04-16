from typing import Any, Dict, List, Tuple
from collections import defaultdict

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class RepoWindowMaker:
    """
    Generates context windows across all Python files in a repository.

    Each window is a symmetric slice of lines around a target line, sampled
    at a regular interval (slice step). These windows are useful for building
    search indexes or training context models.

    Args:
        repo (str): Repository name or relative path (within `data/repositories/`).
        window_size (int): Total number of lines in each context window.
        slice_size (int): Controls how densely windows are sampled across a file.
    """

    def __init__(self, base_dir: str, repo: str, window_size: int, slice_size: int):
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size

        # If slice_size >= window_size, only one window will be created (step = 1)
        self.slice_step = max(1, window_size // slice_size)

        # Dict mapping (tuple-based path) -> file content string
        self.source_code_files: Dict[Tuple[str, ...], str] = Tools.iterate_repository(
            base_dir, repo
        )

    def _build_windows_for_file(
        self, fpath_tuple: Tuple[str, ...], code: str
    ) -> List[Dict[str, Any]]:
        """
        Creates a list of context windows from a single source file.

        Args:
            fpath_tuple: Normalized path to the source file as tuple of path parts.
            code: Raw source code string.

        Returns:
            List of dicts, each representing a code window with metadata.
        """
        code_windows: List[Dict[str, Any]] = []
        code_lines = code.splitlines()
        delta_size = self.window_size // 2

        for line_no in range(0, len(code_lines), self.slice_step):
            start_line = max(0, line_no - delta_size)
            end_line = min(len(code_lines), line_no + self.window_size - delta_size)

            window_lines = [line for line in code_lines[start_line:end_line]]
            if not window_lines:  # skip empty windows
                continue

            context = "\n".join(window_lines)
            metadata = {
                "fpath_tuple": fpath_tuple,
                "line_no": line_no,
                "start_line_no": start_line,
                "end_line_no": end_line,
                "window_size": self.window_size,
                "repo": self.repo,
                "slice_size": self.slice_size,
            }

            code_windows.append(
                {
                    "context": context,
                    "metadata": metadata,
                }
            )

        return code_windows

    def _merge_windows_with_same_context(
        self, code_windows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicates windows by merging metadata of windows with identical context.

        Args:
            code_windows: List of windows (each with 'context' and 'metadata').

        Returns:
            Merged list with unique 'context' entries and grouped metadata.
        """
        merged_code_windows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for window in code_windows:
            context = window["context"]
            merged_code_windows[context].append(window["metadata"])

        return [
            {"context": context, "metadata": metadata_list}
            for context, metadata_list in merged_code_windows.items()
        ]

    def build_windows(self) -> None:
        """
        Builds windows for the entire repository and writes them to a pickle file.
        Each window is a symmetric context slice sampled at regular intervals.
        """
        all_code_windows: List[Dict[str, Any]] = []

        for fpath_tuple, code in self.source_code_files.items():
            all_code_windows.extend(self._build_windows_for_file(fpath_tuple, code))

        merged_windows = self._merge_windows_with_same_context(all_code_windows)

        print(
            f"Built {len(merged_windows)} windows for repo '{self.repo}' "
            f"(window size: {self.window_size}, slice size: {self.slice_size})"
        )

        output_path = FilePathBuilder.repo_windows_path(
            self.repo, self.window_size, self.slice_size
        )
        Tools.dump_pickle(merged_windows, output_path)
