import os

from src.utils.constants import Constants


class FilePathBuilder:
    """
    Utility class for constructing standardized file paths used throughout the pipeline.
    """

    @staticmethod
    def create_dir(file_path: str) -> None:
        """
        Ensures the directory for the given file path exists.
        """
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def repo_windows_path(repo: str, window_size: int, slice_size: int) -> str:
        """
        Constructs the path to store repo context windows.
        """
        out_path = os.path.join(
            f"{Constants.base_cache_windows_dir}/repos/{repo}_ws{window_size}_slice{slice_size}.pkl"
        )
        FilePathBuilder.create_dir(out_path)
        return out_path

    @staticmethod
    def search_first_window_path(
        benchmark: str, mode: str, repo: str, window_size: int, slice_size: int
    ) -> str:
        """
        Constructs the path to store the first-stage windows for search-based tasks.
        """
        out_path = os.path.join(
            f"{Constants.base_cache_windows_dir}/{benchmark}/{mode}/{repo}_ws{window_size}_slice{slice_size}.pkl"
        )
        FilePathBuilder.create_dir(out_path)
        return out_path

    @staticmethod
    def gen_first_window_path(
        benchmark: str,
        mode: str,
        prediction_path: str,
        repo: str,
        window_size: int,
        slice_size: int,
    ) -> str:
        """
        Constructs the output path for generated prediction-based context windows.
        """
        prediction_file_name = os.path.basename(prediction_path).replace(".0.jsonl", "")
        out_path = os.path.join(
            f"{Constants.base_cache_windows_dir}/{benchmark}/{mode}/{prediction_file_name}.{repo}_ws{window_size}_slice{slice_size}.pkl"
        )
        FilePathBuilder.create_dir(out_path)
        return out_path

    @staticmethod
    def one_gram_vector_path(window_file: str) -> str:
        """
        Builds the path for storing 1-gram vectors derived from context windows.
        """
        vector_path = window_file.replace("/window/", "/vector/")
        out_path = vector_path.replace(".pkl", ".one-gram.pkl")
        FilePathBuilder.create_dir(out_path)
        return out_path

    @staticmethod
    def ada002_vector_path(window_file: str) -> str:
        """
        Builds the path for storing Ada-002 embedding vectors derived from context windows.
        """
        vector_path = window_file.replace("/window/", "/vector/")
        out_path = vector_path.replace(".pkl", ".ada002.pkl")
        FilePathBuilder.create_dir(out_path)
        return out_path

    @staticmethod
    def retrieval_results_path(
        query_vector_file: str, repo_vector_file: str, max_top_k: int
    ) -> str:
        """
        Constructs a file path for storing retrieval results between query and repo vectors.
        """
        retrieval_base_dir = os.path.dirname(query_vector_file.replace("/vector/", "/retrieval/"))

        # Strip vector type suffix from filenames
        query_file_name = os.path.basename(query_vector_file)
        if query_file_name.endswith(".one-gram.pkl"):
            query_file_name = query_file_name[: -len(".one-gram.pkl")]
        elif query_file_name.endswith(".ada002.pkl"):
            query_file_name = query_file_name[: -len(".ada002.pkl")]

        repo_file_name = os.path.basename(repo_vector_file)[: -len(".pkl")]

        out_path = os.path.join(
            retrieval_base_dir, f"{query_file_name}.{repo_file_name}.top{max_top_k}.pkl"
        )
        FilePathBuilder.create_dir(out_path)
        return out_path
