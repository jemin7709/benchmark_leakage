import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download

HF_DATASET_REPO = "qyang1021/AIR-Bench-Dataset"
HF_FOUNDATION_DIR = "Foundation"
HF_FOUNDATION_META = "Foundation_meta.json"


def _suffix_from_noise_path(noise_path: str) -> str:
    if not noise_path:
        return "audio"
    return noise_path.replace(".", "").replace("/", "-").replace("mp3", "")[1:]


def _available_choice_letters(item: dict) -> list[str]:
    letters: list[str] = []
    for letter, key in (
        ("A", "choice_a"),
        ("B", "choice_b"),
        ("C", "choice_c"),
        ("D", "choice_d"),
    ):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        letters.append(letter)

    return letters or ["A", "B", "C", "D"]


def _default_eval_path_from_pred_path(pred_path: Path) -> Path:
    output_stem = pred_path.stem.replace("predictions", "evaluation", 1)
    return pred_path.with_name(f"{output_stem}.json")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate_evaluations(
    run_evaluations: list[dict],
    run_prediction_paths: list[str],
    run_evaluation_paths: list[str],
    *,
    n_runs: int,
    seed: int,
) -> dict:
    if not run_evaluations:
        raise ValueError("run_evaluations is empty")

    totals_reference = {
        "total_sum": run_evaluations[0].get("total_sum"),
        "fail_num": run_evaluations[0].get("fail_num"),
    }

    total_corrects: list[float] = [
        float(r.get("total_correct", 0)) for r in run_evaluations
    ]
    total_sum = int(totals_reference["total_sum"] or 0)
    total_accs = [c / total_sum if total_sum else 0.0 for c in total_corrects]

    category_avgs = [float(r.get("category_average", 0.0)) for r in run_evaluations]

    task_ids: set[str] = set()
    for r in run_evaluations:
        task_ids.update((r.get("tasks") or {}).keys())

    tasks_agg: dict[str, dict[str, float | int]] = {}
    for task_id in sorted(task_ids):
        totals = []
        corrects = []
        accs = []
        for r in run_evaluations:
            task = (r.get("tasks") or {}).get(task_id)
            if not task:
                continue
            totals.append(int(task.get("total", 0)))
            corrects.append(float(task.get("correct", 0)))
            accs.append(float(task.get("acc", 0.0)))

        total = totals[0] if totals else 0
        correct_mean, correct_std = _mean_std(corrects)
        acc_mean, acc_std = _mean_std(accs)
        tasks_agg[task_id] = {
            "total": int(total),
            "correct_mean": correct_mean,
            "correct_std": correct_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        }

    category_labels: set[str] = set()
    for r in run_evaluations:
        category_labels.update((r.get("categories") or {}).keys())

    categories_agg: dict[str, dict[str, float | int | str]] = {}
    for label in sorted(category_labels):
        totals = []
        corrects = []
        accs = []
        task_name: str | None = None
        for r in run_evaluations:
            cat = (r.get("categories") or {}).get(label)
            if not cat:
                continue
            totals.append(int(cat.get("total", 0)))
            corrects.append(float(cat.get("correct", 0)))
            accs.append(float(cat.get("acc", 0.0)))
            if task_name is None:
                maybe_task_name = cat.get("task_name")
                if isinstance(maybe_task_name, str):
                    task_name = maybe_task_name

        total = totals[0] if totals else 0
        correct_mean, correct_std = _mean_std(corrects)
        acc_mean, acc_std = _mean_std(accs)
        categories_agg[label] = {
            "task_name": task_name or "",
            "total": int(total),
            "correct_mean": correct_mean,
            "correct_std": correct_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
        }

    total_correct_mean, total_correct_std = _mean_std(total_corrects)
    total_acc_mean, total_acc_std = _mean_std(total_accs)
    category_avg_mean, category_avg_std = _mean_std(category_avgs)

    return {
        "inputs": run_prediction_paths,
        "run_evaluations": run_evaluation_paths,
        "n_runs": int(n_runs),
        "seed": int(seed),
        "tasks": tasks_agg,
        "categories": categories_agg,
        "category_average_mean": category_avg_mean,
        "category_average_std": category_avg_std,
        "total_sum": int(total_sum),
        "total_correct_mean": total_correct_mean,
        "total_correct_std": total_correct_std,
        "total_acc_mean": total_acc_mean,
        "total_acc_std": total_acc_std,
        "fail_num": int(totals_reference["fail_num"] or 0),
    }


def _load_foundation_meta(*, limit: int, max_per_task: int, seed: int) -> list[dict]:
    download_root = Path(
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[f"{HF_FOUNDATION_DIR}/{HF_FOUNDATION_META}"],
            max_workers=2,
            resume_download=True,
        )
    )
    dataset_dir = download_root / HF_FOUNDATION_DIR

    with (dataset_dir / HF_FOUNDATION_META).open("r", encoding="utf-8") as f:
        meta: list[dict] = json.load(f)

    if limit > 0:
        meta = meta[:limit]

    if max_per_task > 0:
        task_to_instances: defaultdict[str, list[dict]] = defaultdict(list)
        for item in meta:
            task_to_instances[str(item.get("task_name", ""))].append(item)

        rng = random.Random(seed)
        sampled_meta: list[dict] = []
        for instances in task_to_instances.values():
            if len(instances) > max_per_task:
                sampled_meta.extend(rng.sample(instances, k=max_per_task))
            else:
                sampled_meta.extend(instances)
        meta = sampled_meta

    return meta


def _write_predictions(meta: list[dict], output_path: Path, rng: random.Random) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for item in meta:
            letters = _available_choice_letters(item)
            response = rng.choice(letters)
            record = {
                "path": item.get("path"),
                "question": item.get("question"),
                "choice_a": item.get("choice_a"),
                "choice_b": item.get("choice_b"),
                "choice_c": item.get("choice_c"),
                "choice_d": item.get("choice_d"),
                "answer_gt": item.get("answer_gt"),
                "task_name": item.get("task_name"),
                "dataset_name": item.get("dataset_name"),
                "response": response,
                "uniq_id": item.get("uniq_id"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_scoring(pred_path: Path, eval_path: Path, *, python_exe: str) -> None:
    cmd = [
        python_exe,
        "src/air-bench/foundation_scoring.py",
        "--input",
        str(pred_path),
        "--output",
        str(eval_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./results/air-bench")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-per-task", type=int, default=3000)
    parser.add_argument(
        "--scoring-python",
        type=str,
        default="",
        help="Path to python executable used for scoring (default: .venv-eval/bin/python if present).",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Generate predictions only (no per-run evaluation or aggregate).",
    )
    args = parser.parse_args()

    if args.num_runs <= 0:
        raise SystemExit("--num-runs must be > 0")

    suffix = _suffix_from_noise_path(args.noise_path)
    output_dir = Path(args.output_dir)

    base_model = f"random_n{args.num_runs}_seed{args.seed}"
    scoring_python = args.scoring_python
    if not scoring_python:
        candidate = Path(".venv-eval/bin/python")
        scoring_python = str(candidate) if candidate.exists() else sys.executable

    print("=" * 100)
    print("AIR-Bench random baseline")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Runs: {args.num_runs}")
    print(f"Seed: {args.seed}")
    print(f"Suffix: {suffix}")
    print(f"Output dir: {output_dir}")
    print(f"Scoring python: {scoring_python}")
    print("=" * 100)

    meta = _load_foundation_meta(
        limit=args.limit, max_per_task=args.max_per_task, seed=args.seed
    )
    print(
        f"Loaded {len(meta)} items from {HF_DATASET_REPO} ({HF_FOUNDATION_META} only)"
    )

    run_prediction_paths: list[str] = []
    run_evaluation_paths: list[str] = []
    run_evaluations: list[dict] = []

    for run_idx in range(args.num_runs):
        run_model = f"{base_model}_run{run_idx:03d}"
        pred_path = output_dir / f"{run_model}_predictions_foundation_{suffix}.jsonl"

        rng = random.Random(args.seed + run_idx)
        t0 = time.time()
        _write_predictions(meta, pred_path, rng)
        dt = time.time() - t0
        print(
            f"[{run_idx + 1:03d}/{args.num_runs:03d}] wrote predictions: {pred_path} ({dt:.2f}s)"
        )

        run_prediction_paths.append(str(pred_path))

        if args.skip_scoring:
            continue

        eval_path = _default_eval_path_from_pred_path(pred_path)
        _run_scoring(pred_path, eval_path, python_exe=scoring_python)
        print(f"[{run_idx + 1:03d}/{args.num_runs:03d}] wrote evaluation:  {eval_path}")

        run_evaluation_paths.append(str(eval_path))
        with eval_path.open("r", encoding="utf-8") as f:
            run_evaluations.append(json.load(f))

    if args.skip_scoring:
        print("Done (predictions only).")
        return

    aggregate = aggregate_evaluations(
        run_evaluations,
        run_prediction_paths,
        run_evaluation_paths,
        n_runs=args.num_runs,
        seed=args.seed,
    )

    aggregate_path = output_dir / f"{base_model}_evaluation_foundation_{suffix}.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")

    print("=" * 100)
    print(f"Done. Wrote aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
