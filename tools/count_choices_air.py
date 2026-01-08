#!/usr/bin/env python3
"""
AIR-Bench jsonl 결과 파일에서 선택지(A, B, C, D) 빈도를 카운트합니다.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_prediction_letter(response: Any) -> str | None:
    if response is None:
        return None

    if not isinstance(response, str):
        response = str(response)

    predict = response.strip().replace("\n", "")
    if not predict or predict == "None":
        return None

    # foundation_scoring.py와 동일한 로직
    first = predict[0]
    if first in {"A", "B", "C", "D"}:
        return first

    if len(predict) > 1:
        maybe = predict[-2]
        if maybe in {"A", "B", "C", "D"}:
            return maybe

    return None


def _ground_truth_letter(record: dict) -> str | None:
    answer_gt = record.get("answer_gt")
    if answer_gt is None:
        return None

    if answer_gt == record.get("choice_a"):
        return "A"
    if answer_gt == record.get("choice_b"):
        return "B"
    if answer_gt == record.get("choice_c"):
        return "C"
    if answer_gt == record.get("choice_d"):
        return "D"

    return None


def count_choices(input_path: Path, task_filter: str | None = None) -> None:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print(f"No records found in {input_path.name}")
        return

    print(f"### File: {input_path.name} ###\n")

    if task_filter:
        tasks = [task_filter]
    else:
        tasks = sorted(list(set(r.get("task_name", "unknown") for r in records)))

    for task in tasks:
        task_records = [r for r in records if r.get("task_name") == task]
        if not task_records:
            continue

        model_counter: Counter[str] = Counter()
        gt_counter: Counter[str] = Counter()
        total = len(task_records)

        for record in task_records:
            response = record.get("response")
            pred = _parse_prediction_letter(response)
            if pred:
                model_counter[pred] += 1
            else:
                model_counter["etc"] += 1

            gt = _ground_truth_letter(record)
            if gt:
                gt_counter[gt] += 1
            else:
                gt_counter["etc"] += 1

        print(f"=== Model Response - {task} ===")
        print(f"Total responses: {total}")
        for choice in sorted(model_counter.keys(), key=lambda x: (x == "etc", x)):
            count = model_counter[choice]
            pct = (count / total * 100) if total > 0 else 0
            print(f"{choice}: {count} ({pct:.1f}%)")
        print()

        print(f"=== Ground Truth - {task} ===")
        print(f"Total responses: {total}")
        for choice in sorted(gt_counter.keys(), key=lambda x: (x == "etc", x)):
            count = gt_counter[choice]
            pct = (count / total * 100) if total > 0 else 0
            print(f"{choice}: {count} ({pct:.1f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="AIR-Bench 선택지 분포 카운트")
    parser.add_argument("path", help="jsonl 파일 또는 폴더 경로")
    parser.add_argument("--task", help="task_name 필터링 (부분 일치)")

    args = parser.parse_args()
    path = Path(args.path).expanduser()

    if path.is_file():
        count_choices(path, args.task)
    elif path.is_dir():
        files = sorted(glob.glob(str(path / "*.jsonl")))
        for i, f in enumerate(files):
            if i > 0:
                print("\n" + "=" * 40 + "\n")
            count_choices(Path(f), args.task)
    else:
        print(f"Error: Path not found {path}")


if __name__ == "__main__":
    main()
