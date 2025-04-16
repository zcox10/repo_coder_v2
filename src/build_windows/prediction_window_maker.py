from typing import Any, Callable, Dict, List

from src.build_windows.base_window_maker import BaseWindowMaker
from src.utils.tools import Tools


class PredictionWindowMaker(BaseWindowMaker):
    """
    Builds context windows by inserting model predictions into the original source code.

    Each window is centered around a prediction line. The predicted text is inserted,
    and then a window of lines is extracted around it. This is used to evaluate how
    inserted completions change the local context.

    Args:
        repo (str): The repository name.
        window_size (int): Total number of lines in the prediction context window.
        prediction_path (str): Path to the `.jsonl` file containing model predictions.
        window_path_builder (Callable): Function that returns an output path based on prediction metadata.
    """

    def __init__(
        self,
        base_dir: str,
        repo: str,
        window_size: int,
        prediction_path: str,
        window_path_builder: Callable[[str, str, int], str],
    ):
        super().__init__(base_dir, repo, window_size)
        self.prediction_path = prediction_path
        self.predictions: List[Dict[str, Any]] = Tools.load_jsonl(prediction_path)
        self.window_path_builder = window_path_builder

    def build_window(self, type: str = "centered") -> None:
        """
        Constructs windows by inserting predicted text at the specified line
        and extracting a symmetric window around the insertion point.

        Args:
            type (str): Currently unused; placeholder for future options.
        """
        code_windows: List[Dict[str, Any]] = []

        for prediction in self.predictions:
            if prediction["metadata"]["task_id"].split("/")[0] != self.repo:
                continue

            fpath_tuple = tuple(prediction["metadata"]["fpath_tuple"])
            line_no: int = prediction["metadata"]["line_no"]
            context_start_lineno: int = prediction["metadata"]["context_start_lineno"]

            code_lines = self._get_code_lines(fpath_tuple)

            # Get all predicted completions (usually just one)
            for sample in [choice["text"] for choice in prediction["choices"]]:
                sample_lines = [line for line in sample.splitlines() if line.strip()]

                # Insert prediction lines into the original source before the predicted line
                new_code_lines = code_lines[:line_no] + sample_lines + code_lines[line_no:]

                # Calculate window bounds *after* inserting prediction
                start_line, end_line = self._get_line_bounds(line_no, context_start_lineno)
                end_line = min(len(new_code_lines), end_line)

                window_lines = [
                    line for line in new_code_lines[start_line:end_line] if line.strip()
                ]
                if not window_lines:
                    continue

                context = "\n".join(window_lines)
                metadata = self._make_metadata(
                    fpath_tuple=fpath_tuple,
                    line_no=line_no,
                    start_line=start_line,
                    end_line=end_line,
                    context_start_lineno=context_start_lineno,
                    task_id=prediction["metadata"]["task_id"],
                    extra_metadata={"prediction": sample},
                )

                code_windows.append(
                    {
                        "context": context,
                        "metadata": metadata,
                    }
                )

        self._log(
            f"Build {len(code_windows)} prediction windows for {self.repo} with window size {self.window_size}"
        )

        output_path = self.window_path_builder(self.prediction_path, self.repo, self.window_size)
        Tools.dump_pickle(code_windows, output_path)
