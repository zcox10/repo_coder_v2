import os
from typing import List, Tuple, Dict, Any, Callable

from src.utils.constants import Constants
from src.utils.tools import Tools


class BuildPrompt:
    """
    Constructs a 2nd-stage prompt by prepending relevant retrieved context blocks
    to the original task prompt.
    """

    def __init__(
        self,
        query_lines_with_retrieval_results: List[Dict[str, Any]],
        task_path: str,
        log_message: str,
        tokenizer: Callable,
        max_retrieval_length: int = 2000,
    ):
        self.query_lines_with_retrieval_results = query_lines_with_retrieval_results
        self.log_message = log_message
        self.tokenizer = tokenizer()
        self.max_retrieval_length = max_retrieval_length

        self.tasks_by_task_id = {
            task["metadata"]["task_id"]: task for task in Tools.load_jsonl(task_path)
        }
        self.separator = "# " + "-" * 50
        self.max_examples = 10

    def _make_a_block(self, retrieved_context: Tuple[Dict[str, Any], float]) -> Tuple[str, int]:
        content, _ = retrieved_context
        metadata = content["metadata"]
        f_paths = ["/".join(x["fpath_tuple"][1:]) for x in metadata]
        f_paths_str = "\n".join([f"# {f_path}" for f_path in f_paths])
        comment_lines = [f"# {line}" for line in content["context"].splitlines()]

        block = "\n".join(
            [
                "# the below code fragment can be found in:",
                f_paths_str,
                self.separator,
                *comment_lines,
                self.separator,
                "",
            ]
        )
        token_len = len(self.tokenizer.tokenize(block))
        return block, token_len

    def _make_an_extended_block(
        self, task_metadata: Dict[str, Any], retrieved_context: Tuple[Dict[str, Any], float]
    ) -> Tuple[str, int]:
        content, _ = retrieved_context
        for meta in content["metadata"]:
            if (
                meta["fpath_tuple"] == tuple(task_metadata["fpath_tuple"])
                and meta["end_line_no"] >= task_metadata["line_no"]
            ):
                continue

            file_path = os.path.join("data/repositories", *meta["fpath_tuple"])
            code_lines = Tools.read_code(file_path).splitlines()
            new_end = min(
                meta["end_line_no"] + meta["window_size"] // meta["slice_size"], len(code_lines)
            )
            new_start = max(0, new_end - meta["window_size"])
            snippet = code_lines[new_start:new_end]
            comment_lines = [f"# {line}" for line in snippet]
            f_paths_str = "\n".join(
                [f"# {'/'.join(x['fpath_tuple'][1:])}" for x in content["metadata"]]
            )

            block = "\n".join(
                [
                    "# the below code fragment can be found in:",
                    f_paths_str,
                    self.separator,
                    *comment_lines,
                    self.separator,
                    "",
                ]
            )
            token_len = len(self.tokenizer.tokenize(block))
            return block, token_len

        return "", 0

    def _build_prompt(
        self,
        mode: str,
        prompt: str,
        task_metadata: Dict[str, Any],
        top_k_context: List[Tuple[Dict[str, Any], float]],
    ) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
        make_block = self._make_an_extended_block if mode == Constants.rg else self._make_a_block
        blocks = []
        current_token_length = 20  # assume fixed prompt head length
        chosen_context = []

        for retrieved_context in reversed(top_k_context):
            if len(chosen_context) >= self.max_examples:
                break
            kwargs = {"retrieved_context": retrieved_context}
            if mode == Constants.rg:
                kwargs["task_metadata"] = task_metadata
            block_str, token_len = make_block(**kwargs)
            if current_token_length + token_len < self.max_retrieval_length:
                blocks.insert(0, block_str)
                current_token_length += token_len
                chosen_context.append(retrieved_context)

        header = (
            "# Here are some relevant code fragments from other files of the repo:\n"
            + self.separator
            + "\n"
        )
        return header + "".join(blocks) + "\n" + prompt, chosen_context

    def build_2nd_stage_input_file(self, mode: str) -> List[Dict[str, Any]]:
        prompts = []
        for query in self.query_lines_with_retrieval_results:
            task_id = query["metadata"]["task_id"]
            task = self.tasks_by_task_id[task_id]
            prompt, context = self._build_prompt(
                mode, task["prompt"], task["metadata"], query["top_k_context"]
            )
            prompts.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        **task["metadata"],
                        "query_window": {
                            "context": query["context"],
                            "metadata": query["metadata"],
                        },
                        "top_k_context": [
                            {
                                "context": c[0]["context"],
                                "metadata": c[0]["metadata"],
                                "sim_score": c[1],
                            }
                            for c in context
                        ],
                        "window_size": query["metadata"]["window_size"],
                        "slice_size": (
                            context[0][0]["metadata"][0]["slice_size"] if context else None
                        ),
                    },
                }
            )

        print("done! " + self.log_message)
        return prompts
