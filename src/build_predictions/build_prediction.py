import os
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.tools import Tools
from src.utils.constants import Constants


class BuildPrediction:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        # self.model.cuda()
        self.batch_size = batch_size
        print("done loading model")

    def _get_batchs(self, prompts, batch_size):
        batches = []
        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i : i + batch_size])
        return batches

    def _generate_batch(self, prompt_batch, max_new_tokens=100):
        prompts = self.tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)

        input_ids = prompts["input_ids"]
        attention_mask = prompts["attention_mask"]

        max_length = self.model.config.max_position_embeddings
        input_lengths = (attention_mask != 0).sum(dim=1)
        adjusted_max_new_tokens = min(max_new_tokens, int(max_length - input_lengths.max().item()))

        print(f"max_length: {max_length}")
        print(f"input_length: {input_lengths.max().item()}")

        if adjusted_max_new_tokens <= 0:
            raise ValueError(
                f"Prompt too long! Cannot generate any new tokens within the context limit ({max_length})"
            )

        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=adjusted_max_new_tokens,
            )

        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for i in range(len(gen_text)):
            gen_text[i] = gen_text[i][len(prompt_batch[i]) :]
        return gen_text

    def batch_generate(self, file):
        print(f"generating from {file}")
        lines = Tools.load_jsonl(file)
        # have a new line at the end
        prompts = [f"{line['prompt']}\n" for line in lines]

        # for prompt in prompts:
        #     print(prompt)
        #     print()
        #     print()
        #     print()

        batches = self._get_batchs(prompts, self.batch_size)
        gen_text = []
        for batch in tqdm.tqdm(batches):
            print(batch)
            gen_text.extend(self._generate_batch(batch))
        print(f"generated {len(gen_text)} samples")
        assert len(gen_text) == len(prompts)
        new_lines = []
        for line, gen in zip(lines, gen_text):
            new_lines.append(
                {
                    "prompt": line["prompt"],
                    "metadata": line["metadata"],
                    "choices": [{"text": gen}],
                }
            )

        # Compose output path: e.g., "rg-one-gram-ws-20-ss-2_samples"
        base_name = os.path.splitext(os.path.basename(file))[0]
        model_suffix = self.model_name.split("/")[-1]  # e.g., "codegen-350M-mono"

        Tools.create_dir(Constants.base_predictions_dir)
        out_file = os.path.join(Constants.base_predictions_dir, f"{base_name}.{model_suffix}.jsonl")
        print(out_file)

        # Save predictions
        # Tools.dump_jsonl(new_lines, out_file)
        # print(f"Saved predictions to {out_file}")
