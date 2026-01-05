import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from huggingface_hub import snapshot_download
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gemma3n import Gemma3n_VLLM  # noqa: E402
from model.qwen2_audio import Qwen2Audio_HF  # noqa: E402
from model.qwen3_omni import Qwen3Omni_VLLM  # noqa: E402
from model.qwen25_omni import Qwen2_5Omni_VLLM  # noqa: E402

HF_DATASET_REPO = "qyang1021/AIR-Bench-Dataset"
HF_FOUNDATION_DIR = "Foundation"

PROMPT_INSTRUCTION = (
    "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, "
    "you may only choose A or B or C or D."
)

MODEL_CLASSES = {
    "gemma3n": Gemma3n_VLLM,
    "qwen3-omni-thinking": lambda: Qwen3Omni_VLLM(
        model_name="Qwen/Qwen3-Omni-30B-A3B-Thinking"
    ),
    "qwen3-omni-instruction": lambda: Qwen3Omni_VLLM(
        model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    ),
    "qwen2.5-omni": Qwen2_5Omni_VLLM,
    "qwen2-audio": Qwen2Audio_HF,
}


def _get_audio_path(dataset_dir: Path, item: dict) -> Path:
    task_name = item["task_name"]
    dataset_name = item["dataset_name"]
    path = item["path"]

    audio_path = dataset_dir / f"{task_name}_{dataset_name}" / path
    if audio_path.exists():
        return audio_path

    if task_name == "Audio_Grounding" and path.endswith(".wav"):
        flac_path = dataset_dir / f"{task_name}_{dataset_name}" / (path[:-3] + "flac")
        if flac_path.exists():
            return flac_path

    raise FileNotFoundError(f"Audio not found: {audio_path}")


def main() -> None:
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
            "qwen2-audio",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--noise-path", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-task", type=int, default=3000)
    args = parser.parse_args()

    set_seed(0)

    print("=" * 100)
    print("Loading AIR-Bench Foundation dataset...")
    dataset_dir = (
        Path(
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                allow_patterns=[f"{HF_FOUNDATION_DIR}/**"],
                max_workers=2,
                resume_download=True,
            )
        )
        / HF_FOUNDATION_DIR
    )
    print(f"Dataset directory: {dataset_dir}")
    print("=" * 100)

    with (dataset_dir / "Foundation_meta.json").open("r", encoding="utf-8") as f:
        meta: list[dict] = json.load(f)

    if args.limit > 0:
        meta = meta[: args.limit]

    if args.max_per_task > 0:
        task_to_instances: defaultdict[str, list[dict]] = defaultdict(list)
        for item in meta:
            task = item["task_name"]
            task_to_instances[task].append(item)

        sampled_meta = []
        for instances in task_to_instances.values():
            if len(instances) > args.max_per_task:
                np.random.seed(0)
                selected_indices = np.random.choice(
                    len(instances),
                    args.max_per_task,
                    replace=False,
                )
                selected_instances = [instances[i] for i in selected_indices]
            else:
                selected_instances = instances
            sampled_meta.extend(selected_instances)

        meta = sampled_meta

    model_name: str = args.model
    noise_path: str = args.noise_path
    model = MODEL_CLASSES[model_name]()

    suffix = (
        noise_path.replace(".", "").replace("/", "-").replace("mp3", "")[1:]
        if noise_path
        else "audio"
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path("./results/air-bench")
        / f"{model_name.replace('.', '')}_predictions_foundation_{suffix}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("AIR-Bench Foundation Inference")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Model: {model_name}")
    print(f"Samples: {len(meta)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_path}")
    print("=" * 100)

    with output_path.open("w", encoding="utf-8") as fout:
        for start in range(0, len(meta), args.batch_size):
            batch_items = meta[start : start + args.batch_size]

            indices: list[int] = []
            prompts: list[str] = []
            audios: list[object] = []
            resolved_audio_paths: list[Path] = []

            for offset, item in enumerate(batch_items):
                idx = start + offset
                indices.append(idx)

                if noise_path:
                    audio_path = Path(noise_path)
                else:
                    audio_path = _get_audio_path(dataset_dir, item)
                audio = librosa.load(str(audio_path), sr=16000)[0]

                choices = f"A. {item['choice_a']}\nB. {item['choice_b']}\nC. {item.get('choice_c', '')}\nD. {item.get('choice_d', '')}"
                prompt = f"{PROMPT_INSTRUCTION}\n{item['question']}\n{choices}"

                prompts.append(prompt)
                audios.append(audio)
                resolved_audio_paths.append(audio_path)

            print("=" * 100)
            print(
                f"Processing batch {start // args.batch_size + 1} "
                f"(size={len(prompts)}) @ {time.strftime('%Y-%m-%d %H:%M:%S')}."
            )
            print("=" * 100)

            batch_responses = model.inference(prompts, audios)

            for idx, item, audio_path, prompt, response in zip(
                indices, batch_items, resolved_audio_paths, prompts, batch_responses
            ):
                print("-" * 100)
                print(
                    f"[{idx}] task={item['task_name']} dataset={item['dataset_name']} uniq_id={item['uniq_id']}"
                )
                print(f"audio_path={audio_path}")
                print("PROMPT:")
                print(prompt)
                print("-" * 100)
                print("OUTPUT:")
                print(response)
                print("=" * 100)

                record = {
                    "path": item["path"],
                    "question": item["question"],
                    "choice_a": item["choice_a"],
                    "choice_b": item["choice_b"],
                    "choice_c": item.get("choice_c"),
                    "choice_d": item.get("choice_d"),
                    "answer_gt": item["answer_gt"],
                    "task_name": item["task_name"],
                    "dataset_name": item["dataset_name"],
                    "response": response,
                    "uniq_id": item["uniq_id"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()
