import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import List, Dict, Any

from src.utils.file_path_builder import FilePathBuilder
from src.utils.tools import Tools


class BagOfWords:
    """
    Vectorizer that tokenizes context windows using a 1-gram (unigram) model
    with the Codex tokenizer. Each context is represented by its token IDs.
    """

    def __init__(self, input_file: str):
        """
        Args:
            input_file (str): Path to a pickle file containing context windows.
        """
        self.input_file = input_file

    def build(self) -> None:
        """
        Builds the 1-gram vectors for the input windows.
        Saves the output as a pickle file where each line includes the context,
        its metadata, and the embedding (token IDs).
        """
        print(f"Building 1-gram vectors for: {self.input_file}")
        lines = Tools.load_pickle(self.input_file)

        new_lines: List[Dict[str, Any]] = []
        futures = {}

        with ProcessPoolExecutor(max_workers=48) as executor:
            for line in lines:
                # Submit tokenization task for each context
                futures[executor.submit(Tools.tokenize, line["context"])] = line

            pbar = tqdm.tqdm(total=len(futures), desc="Tokenizing windows")
            for future in as_completed(futures):
                line = futures[future]
                tokenized = future.result()
                new_lines.append(
                    {
                        "context": line["context"],
                        "metadata": line["metadata"],
                        "data": [{"embedding": tokenized}],
                    }
                )
                pbar.update(1)

        # Dump results to vector file
        output_file_path = FilePathBuilder.one_gram_vector_path(self.input_file)
        Tools.dump_pickle(new_lines, output_file_path)
        print(f"Saved vectors to: {output_file_path}")
