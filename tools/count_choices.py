#!/usr/bin/env python3
"""
model_response에서 선택지(A, B, C, D...) 빈도를 카운트합니다.
A., B. 패턴이 없는 응답은 etc로 집계됩니다.
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def load_parquet(parquet_path: str, category: str) -> pd.DataFrame:
    if not parquet_path.endswith(".parquet"):
        raise ValueError(f"Expected .parquet file, got: {parquet_path}")

    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    return cast(pd.DataFrame, df[df["category"] == category].reset_index(drop=True))


def normalize_to_list(choices: Any) -> list[Any] | None:
    if choices is None:
        return None
    if isinstance(choices, (list, tuple, np.ndarray, pd.Series)):
        return list(choices)
    return None


def find_matching_index_exact(answer: Any, choices: list[Any]) -> int:
    if answer in choices:
        return choices.index(answer)
    return -1


def find_matching_index_substring(answer: str, choices: list[Any]) -> int:
    a_low = answer.lower().strip()
    for i, choice in enumerate(choices):
        if not isinstance(choice, str):
            continue
        c_low = choice.lower().strip()
        if a_low in c_low or c_low in a_low:
            return i
    return -1


def find_matching_index_wordset(answer: str, choices: list[Any]) -> int:
    a_words = set(re.findall(r"\w+", answer.lower().strip()))
    for i, choice in enumerate(choices):
        if not isinstance(choice, str):
            continue
        c_words = set(re.findall(r"\w+", choice.lower().strip()))
        if a_words and c_words and (a_words <= c_words or c_words <= a_words):
            return i
    return -1


def find_matching_index(answer: Any, choices: list[Any] | None) -> int:
    if choices is None:
        return -1

    exact_match = find_matching_index_exact(answer, choices)
    if exact_match != -1:
        return exact_match

    if not isinstance(answer, str):
        return -1

    substring_match = find_matching_index_substring(answer, choices)
    if substring_match != -1:
        return substring_match

    return find_matching_index_wordset(answer, choices)


def index_to_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def count_model_responses(df: pd.DataFrame, response_col: str) -> Counter[str]:
    counter: Counter[str] = Counter()

    for response in df[response_col]:
        if response is None or pd.isna(response):
            counter["etc"] += 1
            continue

        matches = re.findall(r"[A-Z]\.", str(response))
        if matches:
            for m in matches:
                counter[m[0]] += 1
        else:
            counter["etc"] += 1

    return counter


def count_ground_truth(df: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()

    for _, row in df.iterrows():
        choices = normalize_to_list(row.get("choices"))
        answer = row.get("answer")
        matched_idx = find_matching_index(answer, choices)

        if matched_idx != -1:
            counter[index_to_letter(matched_idx)] += 1
        else:
            counter["etc"] += 1

    return counter


def format_counter(counter: Counter[str], total: int) -> list[tuple[str, int, float]]:
    sorted_keys = sorted(counter.keys(), key=lambda x: (x == "etc", x))
    return [
        (choice, counter[choice], counter[choice] / total * 100 if total > 0 else 0)
        for choice in sorted_keys
    ]


def output_cli(title: str, counter: Counter[str], total: int) -> None:
    print(f"=== {title} ===")
    print(f"Total responses: {total}")

    for choice, count, pct in format_counter(counter, total):
        print(f"{choice}: {count} ({pct:.1f}%)")


def output_etc_details(df: pd.DataFrame) -> None:
    etc_count = 0

    for idx, row in df.iterrows():
        choices = normalize_to_list(row.get("choices"))
        answer = row.get("answer")

        if find_matching_index(answer, choices) == -1:
            etc_count += 1
            print(f"idx: {idx}")
            print(f"answer: {repr(answer)}")
            print(f"choices: {choices}")
            print()

    print(f"Total etc: {etc_count}")


def output_csv(title: str, counter: Counter[str], total: int, output_path: str) -> None:
    rows = [
        {
            "category": title,
            "choice": choice,
            "count": count,
            "percentage": round(pct, 2),
        }
        for choice, count, pct in format_counter(counter, total)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="model_response와 실제 정답(answer)의 선택지 분포를 함께 출력합니다."
    )
    parser.add_argument(
        "parquet_file",
        help="parquet 파일 경로 (예: results/qwen.parquet)",
    )
    parser.add_argument(
        "--category", required=True, help="필터링할 카테고리 (예: sound, speech, music)"
    )
    parser.add_argument(
        "--response-col", default="model_response", help="model_response 컬럼명"
    )
    parser.add_argument(
        "--output-mode",
        choices=["cli", "csv"],
        default="cli",
        help="출력 모드: cli (기본) 또는 csv",
    )
    parser.add_argument(
        "--output-path", default="./choice_counts.csv", help="CSV 출력 경로"
    )
    args = parser.parse_args()

    try:
        parquet_path = str(Path(args.parquet_file).expanduser())
        df = load_parquet(parquet_path, args.category)
        total = len(df)

        model_counter = count_model_responses(df, args.response_col)
        gt_counter = count_ground_truth(df)

        if args.output_mode == "cli":
            output_cli(f"Model Response - {args.category}", model_counter, total)
            print()
            output_cli(f"Ground Truth - {args.category}", gt_counter, total)
            print()
            print("=== ETC Details (정답이 choices에 없는 경우) ===")
            output_etc_details(df)
        else:
            output_csv(
                f"Model Response - {args.category}",
                model_counter,
                total,
                args.output_path,
            )
            output_csv(
                f"Ground Truth - {args.category}", gt_counter, total, args.output_path
            )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
