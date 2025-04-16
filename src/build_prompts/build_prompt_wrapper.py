import functools
from typing import List, Callable

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.build_prompts.build_prompt import BuildPrompt


class BuildPromptWrapper:
    """
    Wrapper class for generating second-stage prompts from retrieval results and code windows.
    """

    def __init__(
        self,
        vectorizer: str,
        benchmark: str,
        repos: List[str],
        window_size: int,
        slice_size: int,
        tokenizer: Callable,
    ):
        self.vector_path_builder = {
            "one-gram": FilePathBuilder.one_gram_vector_path,
            "ada002": FilePathBuilder.ada002_vector_path,
        }[vectorizer]

        self.benchmark = benchmark
        self.repos = repos
        self.window_size = window_size
        self.slice_size = slice_size
        self.tokenizer = tokenizer

        self.task_path = {
            Constants.line_benchmark: Constants.random_line_completion_benchmark,
            Constants.api_benchmark: Constants.api_completion_benchmark,
            Constants.short_api_benchmark: Constants.short_api_completion_benchmark,
            Constants.short_line_benchmark: Constants.short_random_line_completion_benchmark,
        }[benchmark]

        self.max_top_k = 20

    def _run(self, mode: str, query_window_path_builder: Callable, output_file_path: str) -> None:
        lines = []
        for repo in self.repos:
            query_window_path = query_window_path_builder(repo, self.window_size, self.slice_size)
            query_line_path = self.vector_path_builder(query_window_path)
            repo_window_path = FilePathBuilder.repo_windows_path(
                repo, self.window_size, self.slice_size
            )
            repo_vector_path = self.vector_path_builder(repo_window_path)
            retrieval_path = FilePathBuilder.retrieval_results_path(
                query_line_path, repo_vector_path, self.max_top_k
            )
            query_lines = Tools.load_pickle(retrieval_path)

            builder = BuildPrompt(
                query_lines,
                self.task_path,
                f"repo: {repo}, window: {self.window_size}, slice: {self.slice_size}",
                self.tokenizer,
            )
            lines.extend(builder.build_2nd_stage_input_file(mode))

        Tools.dump_jsonl(lines, output_file_path)

    def build_first_search_prompt(self, mode: str, output_path: str) -> None:
        query_path_fn = functools.partial(
            FilePathBuilder.search_first_window_path, self.benchmark, mode
        )
        self._run(mode, query_path_fn, output_path)

    def build_prediction_prompt(self, mode: str, prediction_path: str, output_path: str) -> None:
        query_path_fn = functools.partial(
            FilePathBuilder.gen_first_window_path, self.benchmark, mode, prediction_path
        )
        self._run(mode, query_path_fn, output_path)
