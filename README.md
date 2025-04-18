# RepoCoder Pipeline (Extended)

This project is inspired by [Microsoft’s RepoCoder](https://github.com/microsoft/CodeT) — a retrieval-augmented framework for repository-level code completion.

> **Citation**  
>
> ```bibtex
> @article{zhang2023repocoder,
>   title={RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation},
>   author={Zhang, Fengji and Chen, Bei and Zhang, Yue and Liu, Jin and Zan, Daoguang and Mao, Yi and Lou, Jian-Guang and Chen, Weizhu},
>   journal={arXiv preprint arXiv:2303.12570},
>   year={2023}
> }
> ```

This extended version provides a modular, fully-automated pipeline for evaluating repository-level code completion with the RepoCoder methodology. The pipeline supports three key retrieval paradigms:

- **RG1 (Retrieve-and-Generate):** Uses top-k retrieval from the repository as context for completing a masked target.
- **GT (Ground Truth):** Uses the oracle context centered around the target as a reference baseline.
- **RG-RG (RepoCoder):** A second round of retrieval using model predictions inserted into the source to regenerate prompts and improve completions.

## Conceptual Overview

The goal is to evaluate how retrieval can enhance long-range code completion at the repository level. The process revolves around constructing and manipulating **context windows**—chunks of code extracted from repositories—which are later used to retrieve similar code segments and construct prompts for language models.

### Step 1: Repository Windowing

To build a retrieval corpus, the pipeline first slices every Python file in each repository into **sliding context windows**. A window is a fixed number of lines (e.g., 20), sampled every few lines (e.g., every 2 lines). These windows are saved with metadata (file name, line numbers, etc.) and used as the searchable database for later retrieval.

This windowing is performed **agnostic to any task**, purely to index the repo structure.

Purpose:

- Create a dense, searchable corpus of code chunks for similarity retrieval.
- Represent all possible surrounding contexts in the codebase.

### Step 2: Vectorization

Each context window is then converted into a **vector representation**. This repo supports:

- **Bag-of-Words (1-gram)**: A simple token-based representation based on the number of unigrams.
- (Future extensibility exists for embedding models like `text-embedding-ada-002`.)

Purpose:

- Translate raw code into a numerical format that allows similarity comparison.
- Facilitate fast nearest-neighbor search during retrieval.

These vectors are saved and reused for each retrieval strategy.

### Step 3: Task-Based Windowing (RG1 and GT)

A benchmark file (e.g., `short_api_benchmark`) defines **specific locations in the code** where a completion should be evaluated. For each location:

- **RG1 (Retrieve-and-Generate)** extracts a one-sided window that ends at the target line. This simulates a real-world scenario where the model completes code after reading what came before.
  
- **GT (Ground Truth)** extracts a symmetric window centered on the target line, treating it as a reference for comparison. This reflects an ideal oracle context.

Purpose:

- RG1 serves as the **query** for retrieval.
- GT serves as the **gold standard** for performance evaluation.

These task-specific windows are later used to build prompts.

### Step 4: Retrieval

Using the vectorized task windows (from RG1 and GT), the pipeline performs **nearest-neighbor search** against the repository-wide window vectors.

- Top-k similar code fragments are retrieved.
- These fragments act as **in-context examples** for the prompt.

Purpose:

- Leverage structurally and semantically similar code as reference material for the model.
- Emulate how developers might "look around the repo" for patterns.

### Step 5: Prompt Construction

The retrieved fragments are formatted as **commented code blocks** with file path annotations and included above the original prompt.

A prompt typically looks like:

```bash
the below code fragment can be found in:
utils.py
--------------------------------------------------
def load_json(file_path):
with open(file_path) as f:
return json.load(f)
--------------------------------------------------
Here is some code to complete...
```

- Length is constrained by model context size (e.g., 2048 tokens).
- Multiple fragments are included if space allows.

**Purpose:**

- Mimic few-shot examples using real repository context.
- Improve generation by grounding the model in relevant patterns.

### Step 6: Inference (External to This Pipeline)

The `.jsonl` files of prompts are passed to a model such as **CodeGen**, **Codex**, or **GPT-4**.

- The model generates completions conditioned on the prompt.
- Output format should include the generated `choices` and match the benchmark structure.

This step must be run separately using the model of your choice.

### Step 7: Prediction-Based Retrieval (RepoCoder / RG-RG)

This is the **key innovation** introduced by RepoCoder:

- After inference, model predictions are inserted back into the source code at the original locations.
- New context windows are sliced around these predictions.
- These windows are vectorized and used to **re-run retrieval** and **rebuild prompts** (as in Steps 2–5).

This "retrieve-then-generate-then-retrieve-again" loop allows the model to refine its completion by attending to **context that was not visible during the first generation**.

Purpose:

- Boost completion quality with second-stage retrieval tailored to the model's own prediction.
- Simulate an iterative reasoning process.

## Execution Flow in `run.py`

The main driver script runs the following steps:

```python
make_repo_windows(...)            # Step 1: Build and vectorize full-repo sliding windows

run_rg1_and_gt_stage(...)         # Step 2–5: Build RG1 and GT windows, retrieve, construct prompts

run_repocoder_stage(...)          # Step 6–7: Insert predictions, re-vectorize, re-retrieve, rebuild prompts
```

These steps output:

- Vector files (.pkl)
- Retrieval results (.pkl)
- Final prompts for inference (.jsonl)

From there, evaluation proceeds by running completions and scoring them against ground truth.
