import numpy as np
import copy

from src.utils.tools import Tools


class CodeSearchWorker:
    def __init__(
        self,
        repo_embedding_lines,
        query_embedding_lines,
        output_path,
        sim_scorer,
        max_top_k,
        log_message,
    ):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_embedding_lines = query_embedding_lines  # list
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        self.output_path = output_path
        self.log_message = log_message

    def _is_context_after_hole(self, repo_embedding_line, query_line):
        hole_fpath_tuple = tuple(query_line["metadata"]["fpath_tuple"])
        context_is_not_after_hole = []
        for metadata in repo_embedding_line["metadata"]:
            if tuple(metadata["fpath_tuple"]) != hole_fpath_tuple:
                context_is_not_after_hole.append(True)
                continue
            # now we know that the repo line is in the same file as the hole
            if metadata["end_line_no"] <= query_line["metadata"]["context_start_lineno"]:
                context_is_not_after_hole.append(True)
                continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)

    def _find_top_k_context(self, query_line):
        top_k_context = []
        query_embedding = np.array(query_line["data"][0]["embedding"])
        for repo_embedding_line in self.repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line["data"][0]["embedding"])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k :]
        return top_k_context

    def run(self):
        query_lines_with_retrieved_results = []
        for query_line in self.query_embedding_lines:
            new_line = copy.deepcopy(query_line)
            top_k_context = self._find_top_k_context(new_line)
            new_line["top_k_context"] = top_k_context
            query_lines_with_retrieved_results.append(new_line)
        Tools.dump_pickle(query_lines_with_retrieved_results, self.output_path)
