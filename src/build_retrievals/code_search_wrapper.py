# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from concurrent.futures import as_completed, ProcessPoolExecutor
import tqdm
import os
import functools

from src.utils.constants import Constants
from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools
from src.build_retrievals.similarity import SimilarityScore
from src.build_retrievals.code_search_worker import CodeSearchWorker


class CodeSearchWrapper:
    def __init__(self, vectorizer, benchmark, repos, window_sizes, slice_sizes):
        self.vectorizer = vectorizer
        if vectorizer == "one-gram":
            self.sim_scorer = SimilarityScore.jaccard_similarity
            self.vector_path_builder = FilePathBuilder.one_gram_vector_path
        elif vectorizer == "ada002":
            self.sim_scorer = SimilarityScore.cosine_similarity
            self.vector_path_builder = FilePathBuilder.ada002_vector_path
        self.max_top_k = 20  # store 20 top k context for the prompt construction (top 10)
        self.repos = repos
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes
        self.benchmark = benchmark

    def _run_parallel(self, query_window_path_builder, prediction_path_template=None):
        workers = []
        for window_size in self.window_sizes:
            for slice_size in self.slice_sizes:
                for repo in self.repos:
                    if prediction_path_template:
                        query_window_path = query_window_path_builder(
                            prediction_path_template.format(
                                window_size=window_size, slice_size=slice_size
                            ),
                            repo,
                            window_size,
                            slice_size,
                        )
                    else:
                        query_window_path = query_window_path_builder(repo, window_size, slice_size)
                    query_line_path = self.vector_path_builder(query_window_path)
                    repo_window_path = FilePathBuilder.repo_windows_path(
                        repo, window_size, slice_size
                    )
                    repo_embedding_path = self.vector_path_builder(repo_window_path)
                    output_path = FilePathBuilder.retrieval_results_path(
                        query_line_path, repo_embedding_path, self.max_top_k
                    )
                    repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
                    query_embedding_lines = Tools.load_pickle(query_line_path)
                    log_message = f"repo: {repo}, window: {window_size}, slice: {slice_size}  {self.vectorizer}, max_top_k: {self.max_top_k}"
                    worker = CodeSearchWorker(
                        repo_embedding_lines,
                        query_embedding_lines,
                        output_path,
                        self.sim_scorer,
                        self.max_top_k,
                        log_message,
                    )
                    workers.append(worker)
        # process pool
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(
                    worker.run,
                )
                for worker in workers
            }
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

    def search_baseline_and_ground(self):
        query_line_path_temp = functools.partial(
            FilePathBuilder.search_first_window_path, self.benchmark, Constants.rg
        )
        self._run_parallel(query_line_path_temp)
        query_line_path_temp = functools.partial(
            FilePathBuilder.search_first_window_path, self.benchmark, Constants.gt
        )
        self._run_parallel(query_line_path_temp)

    def search_prediction(self, mode, prediction_path_template):
        query_line_path_temp = functools.partial(
            FilePathBuilder.gen_first_window_path, self.benchmark, mode
        )
        self._run_parallel(query_line_path_temp, prediction_path_template)
