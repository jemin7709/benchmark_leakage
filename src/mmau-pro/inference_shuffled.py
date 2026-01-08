import argparse
import glob
import os
import random
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gemma3n import Gemma3n_VLLM
from model.qwen3_omni import Qwen3Omni_VLLM
from model.qwen25_omni import Qwen2_5Omni_VLLM


warnings.filterwarnings("ignore", category=UserWarning)


def load_data() -> tuple[pd.DataFrame, str]:
    data_root = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots"
    )
    parquet_dirs = glob.glob(os.path.join(data_root, "*", "test.parquet"))
    return pd.read_parquet(parquet_dirs[0]), os.path.dirname(parquet_dirs[0])


def shuffle_choices(choices, seed, sample_idx):
    if (
        choices is None
        or not isinstance(choices, (list, np.ndarray))
        or len(choices) <= 1
    ):
        return choices, list(range(len(choices))) if choices is not None else []

    rng = random.Random(seed + sample_idx)

    indices = list(range(len(choices)))
    rng.shuffle(indices)

    shuffled_choices = [choices[i] for i in indices]

    choice_mapping = [0] * len(choices)
    for new_pos, old_pos in enumerate(indices):
        choice_mapping[old_pos] = new_pos

    return shuffled_choices, choice_mapping


def make_prompt(category: str, question: str, choices: list[str]) -> str:
    mcq_prompt = (
        "Choose the most suitable answer from options to respond the question in next line, "
        "you may only choose one option and must write your answer as 'A. answer'.\n"
    )
    open_prompt = "Answer the following question.\n"
    prompt = [
        mcq_prompt
        if category.lower() != "open" and category.lower() != "instruction following"
        else open_prompt
    ]
    prompt.append(question + "\n")
    if category.lower() != "open" and category.lower() != "instruction following":
        choices_list = [
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)
        ]
        prompt.append("\n".join(choices_list))
    return "".join(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3n",
        choices=[
            "gemma3n",
            "qwen3-omni-thinking",
            "qwen3-omni-instruction",
            "qwen2.5-omni",
        ],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--noise-path",
        type=str,
        default="",
        help="Path to the white noise audio file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=0, help="Seed for shuffling choices"
    )
    args = parser.parse_args()
    set_seed(0)

    batch_size: int = args.batch_size
    model_name = args.model
    noise_path = args.noise_path
    shuffle_seed = args.shuffle_seed

    df, data_dir = load_data()
    if model_name == "gemma3n":
        model = Gemma3n_VLLM()
    elif model_name == "qwen3-omni-thinking":
        model = Qwen3Omni_VLLM(model_name="Qwen/Qwen3-Omni-30B-A3B-Thinking")
    elif model_name == "qwen3-omni-instruction":
        model = Qwen3Omni_VLLM(model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    elif model_name == "qwen2.5-omni":
        model = Qwen2_5Omni_VLLM()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    all_responses = []
    df["original_choices"] = None
    df["choice_mapping"] = None
    df["shuffle_seed"] = shuffle_seed

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        prompts = []
        audios = []

        for idx_in_batch, (_, row) in enumerate(batch_df.iterrows()):
            sample_idx = i + idx_in_batch

            category = row["category"]
            choices = row["choices"]

            if (
                category.lower() != "open"
                and category.lower() != "instruction following"
            ):
                shuffled, mapping = shuffle_choices(choices, shuffle_seed, sample_idx)
                df.at[sample_idx, "original_choices"] = choices
                df.at[sample_idx, "choices"] = shuffled
                df.at[sample_idx, "choice_mapping"] = mapping
                current_choices = shuffled
            else:
                current_choices = choices

            prompt = make_prompt(category, row["question"], current_choices)
            prompts.append(prompt)

            if noise_path == "":
                audio_path = os.path.join(data_dir, row["audio_path"][0])
            else:
                audio_path = noise_path
            audios.append(librosa.load(audio_path, sr=16000)[0])

        print(f"Processing batch: {i // batch_size + 1} (size: {len(prompts)})")
        batch_responses = model.inference(prompts, audios)

        all_responses.extend(batch_responses)

        for idx, res in enumerate(batch_responses):
            sample_idx = i + idx
            print(f"Sample {sample_idx}:\n{prompts[idx]}")
            print("-" * 50)
            print(f"Output:\n{res}")
            print("=" * 50)
            df.at[sample_idx, "model_response"] = res

        noise_suffix = (
            noise_path.replace(".", "").replace("/", "-").replace("mp3", "")[1:]
            if noise_path != ""
            else "audio"
        )
        suffix = f"shuffled_seed{shuffle_seed}_{noise_suffix}"

        os.makedirs("./results/mmau-pro", exist_ok=True)
        df.to_parquet(
            os.path.join(
                "./results/mmau-pro",
                f"{model_name.replace('.', '')}_predictions_{suffix}.parquet",
            )
        )
    print(f"Total processed: {len(all_responses)}")
