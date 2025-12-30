import argparse
import json
import os
import sys
import time
from pathlib import Path

import librosa
from huggingface_hub import hf_hub_download
from transformers import set_seed

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.gemma3n import Gemma3n_VLLM  # noqa: E402
from model.qwen3_omni import Qwen3Omni_VLLM  # noqa: E402
from model.qwen25_omni import Qwen2_5Omni_VLLM  # noqa: E402

HF_DATASET_REPO = "qyang1021/AIR-Bench-Dataset"
HF_FOUNDATION_DIR = "Foundation"

PROMPT_INSTRUCTION = (
    "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, "
    "you may only choose A or B or C or D."
)


def _load_meta() -> list[dict]:
    meta_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=f"{HF_FOUNDATION_DIR}/Foundation_meta.json",
    )
    with Path(meta_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(
    question: str, choice_a: str, choice_b: str, choice_c: object, choice_d: object
) -> str:
    choices = f"A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}"
    return PROMPT_INSTRUCTION + "\n" + question + "\n" + choices


def _candidate_audio_paths(item: dict) -> list[str]:
    path = item["path"]
    if item.get("task_name") == "Audio_Grounding" and path.endswith(".wav"):
        return [path, path[:-3] + "flac"]
    return [path]


def _download_audio_if_needed(item: dict) -> tuple[Path, str]:
    task_name = item["task_name"]
    dataset_name = item["dataset_name"]

    last_error: Exception | None = None
    for rel_path in _candidate_audio_paths(item):
        try:
            local_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                filename=f"{HF_FOUNDATION_DIR}/{task_name}_{dataset_name}/{rel_path}",
            )
            return Path(local_path), rel_path
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to download audio for task={task_name} dataset={dataset_name} path={item['path']}: {last_error}"
    )


def _build_model(model_name: str):
    if model_name == "gemma3n":
        return Gemma3n_VLLM()
    if model_name == "qwen3-omni":
        return Qwen3Omni_VLLM()
    if model_name == "qwen2.5-omni":
        return Qwen2_5Omni_VLLM()
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3n",
        choices=["gemma3n", "qwen3-omni", "qwen2.5-omni"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output jsonl path (defaults to ./results/air-bench/{model}_predictions_foundation.jsonl)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If set, only process first N samples (debugging)",
    )
    args = parser.parse_args()

    set_seed(0)

    meta = _load_meta()
    if args.limit and args.limit > 0:
        meta = meta[: args.limit]

    model_name: str = args.model
    model = _build_model(model_name)

    output_path = (
        Path(args.output)
        if args.output
        else Path("./results/air-bench")
        / f"{model_name.replace('.', '')}_predictions_foundation.jsonl"
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

                local_audio_path, _ = _download_audio_if_needed(item)
                audio = librosa.load(str(local_audio_path), sr=None)[0]

                prompt = _build_prompt(
                    question=item["question"],
                    choice_a=item["choice_a"],
                    choice_b=item["choice_b"],
                    choice_c=item.get("choice_c", None),
                    choice_d=item.get("choice_d", None),
                )

                prompts.append(prompt)
                audios.append(audio)
                resolved_audio_paths.append(local_audio_path)

            print("=" * 100)
            print(
                f"Processing batch {start // args.batch_size + 1} "
                f"(size={len(prompts)}) @ {time.strftime('%Y-%m-%d %H:%M:%S')}."
            )
            print("=" * 100)

            batch_responses = model.inference(prompts, audios)

            for idx, item, local_audio_path, prompt, response in zip(
                indices, batch_items, resolved_audio_paths, prompts, batch_responses
            ):
                print("-" * 100)
                print(
                    f"[{idx}] task={item.get('task_name')} dataset={item.get('dataset_name')} uniq_id={item.get('uniq_id')}"
                )
                print(f"audio_path={local_audio_path}")
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
                    "choice_c": item.get("choice_c", None),
                    "choice_d": item.get("choice_d", None),
                    "answer_gt": item.get("answer_gt"),
                    "task_name": item.get("task_name"),
                    "dataset_name": item.get("dataset_name"),
                    "response": response,
                    "uniq_id": item.get("uniq_id"),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    main()
