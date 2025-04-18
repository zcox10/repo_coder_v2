from src.build_predictions.build_prediction import BuildPrediction


def build_predictions() -> None:
    """ """
    file_path = "data/prompts/r-g-one-gram-ws-20-ss-2.jsonl"
    tiny_codegen = "Salesforce/codegen-350M-mono"

    cg = BuildPrediction(tiny_codegen, batch_size=1)
    cg.batch_generate(file_path)
