from typing import List
from transformers import AutoTokenizer

from src.utils.constants import Constants


class CodeGenTokenizer:
    """
    Tokenizer wrapper for CodeGen using HuggingFace Transformers.
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(Constants.codegen_tokenizer)

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes the given text using CodeGen tokenizer.
        """
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back to text.
        """
        return self.tokenizer.decode(token_ids)
