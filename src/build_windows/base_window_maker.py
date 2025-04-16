from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from src.utils.tools import Tools


class BaseWindowMaker(ABC):
    """
    Abstract base class for generating context windows over source code files.
    Handles common functionality such as file loading, context slicing, and dumping output.
    Subclasses need to implement `build_window` with task-specific logic.
    """

    def __init__(self, base_dir: str, repo: str, window_size: int):
        self.base_dir = base_dir
        self.repo = repo
        self.window_size = window_size
        self.delta_size = window_size // 2
        self.source_code: Dict[Tuple[str, ...], str] = Tools.iterate_repository(base_dir, repo)

    @abstractmethod
    def build_window(self) -> None:
        """
        Builds context windows and saves them to disk.
        Implemented by subclasses.
        """
        pass

    def _get_code_lines(self, fpath_tuple: Tuple[str, ...]) -> List[str]:
        """
        Returns the lines of code for a given file path tuple.
        """
        return self.source_code[fpath_tuple].splitlines()

    def _get_context_window(self, code_lines: List[str], start_line: int, end_line: int) -> str:
        """
        Returns a joined string of code lines in the [start_line, end_line) range.
        Strips trailing/leading whitespace lines.
        """
        window_lines = [line for line in code_lines[start_line:end_line] if line.strip()]
        return "\n".join(window_lines)

    def _get_line_bounds(self, line_no: int, context_start_lineno: int = 0) -> Tuple[int, int]:
        """
        Computes start and end bounds for a symmetric context window centered on `line_no`.
        Ensures bounds stay within file limits.
        """
        start_line = max(context_start_lineno, line_no - self.delta_size)
        end_line = line_no + self.window_size - self.delta_size
        return start_line, end_line

    def _make_metadata(
        self,
        fpath_tuple: Tuple[str, ...],
        line_no: int,
        start_line: int,
        end_line: int,
        context_start_lineno: int = 0,
        task_id: str = "",
        extra_metadata: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Standard metadata structure attached to each code window.
        """
        base_metadata = {
            "fpath_tuple": fpath_tuple,
            "line_no": line_no,
            "start_line_no": start_line,
            "end_line_no": end_line,
            "window_size": self.window_size,
            "context_start_lineno": context_start_lineno,
            "repo": self.repo,
        }
        if task_id:
            base_metadata["task_id"] = task_id
        base_metadata.update(extra_metadata)
        return base_metadata

    def _log(self, message: str) -> None:
        """
        Simple logger function.
        """
        print(message)
