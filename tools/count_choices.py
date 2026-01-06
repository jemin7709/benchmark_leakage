#!/usr/bin/env python3
"""
model_response에서 선택지(A, B, C, D...) 빈도를 카운트합니다.
A., B. 패턴이 없는 응답은 etc로 집계됩니다.
--ground-truth 옵션으로 실제 정답 분포도 확인할 수 있습니다.
"""

import argparse
import glob
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def count_choices(
    parquet_dir: str,
    category: str,
    response_col: str,
    include_ground_truth: bool = False,
) -> tuple[dict, int]:
    """선택지 카운트 및 총 응답 수 반환"""
    # 단일 파일인지 디렉토리인지 확인
    if Path(parquet_dir).is_file() and parquet_dir.endswith(".parquet"):
        parquet_files = [parquet_dir]
    else:
        parquet_files = glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)
        if not parquet_files:
            parquet_files = glob.glob(f"{parquet_dir}/*.parquet")

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["category"] == category]

    counter = Counter()
    for _, row in df.iterrows():
        if include_ground_truth:
            # 실제 정답 분포 계산
            choices = row.get("choices")
            answer = row.get("answer")
            if choices is not None and isinstance(
                choices, (list, tuple, np.ndarray, pd.Series)
            ):
                choices = list(choices)

            matched_idx = -1
            if choices is not None and isinstance(choices, list):
                # 1. Exact match
                if answer in choices:
                    matched_idx = choices.index(answer)
                # 2. Flexible match (substring or word-set subset)
                elif isinstance(answer, str):
                    a_low = answer.lower().strip()
                    a_words = set(re.findall(r"\w+", a_low))

                    for i, choice in enumerate(choices):
                        if not isinstance(choice, str):
                            continue
                        c_low = choice.lower().strip()
                        c_words = set(re.findall(r"\w+", c_low))

                        # Substring match
                        if a_low in c_low or c_low in a_low:
                            matched_idx = i
                            break
                        # Word-set subset match (e.g., "It occurs in the beginning" <-> "It occurs only once in the beginning")
                        if (
                            a_words
                            and c_words
                            and (a_words <= c_words or c_words <= a_words)
                        ):
                            matched_idx = i
                            break

            if matched_idx != -1:
                choice_letter = chr(ord("A") + matched_idx)
                counter[choice_letter] += 1
            else:
                counter["etc"] += 1
        else:
            response = row.get(response_col)
            if response is None or (hasattr(pd, "isna") and pd.isna(response)):
                counter["etc"] += 1
                continue

            response_str = str(response)
            matches = re.findall(r"[A-Z]\.", response_str)

            if matches:
                for m in matches:
                    counter[m[0]] += 1
            else:
                counter["etc"] += 1

    return counter, len(df)


def output_cli(category: str, counter: dict, total: int) -> None:
    """CLI 출력"""
    print(f"=== {category} ===")
    print(f"Total responses: {total}")

    sorted_choices = sorted(counter.keys(), key=lambda x: (x == "etc", x))
    for choice in sorted_choices:
        count = counter[choice]
        pct = count / total * 100 if total > 0 else 0
        print(f"{choice}: {count} ({pct:.1f}%)")


def output_etc_details(parquet_dir: str, category: str, response_col: str) -> None:
    """정답이 choices에 없는 경우 상세 정보 출력"""
    if Path(parquet_dir).is_file() and parquet_dir.endswith(".parquet"):
        parquet_files = [parquet_dir]
    else:
        parquet_files = glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)
        if not parquet_files:
            parquet_files = glob.glob(f"{parquet_dir}/*.parquet")

    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["category"] == category]

    etc_count = 0
    for idx, row in df.iterrows():
        choices = row.get("choices")
        answer = row.get("answer")
        if choices is not None and isinstance(
            choices, (list, tuple, np.ndarray, pd.Series)
        ):
            choices = list(choices)

        matched_idx = -1
        if choices is not None and isinstance(choices, list):
            if answer in choices:
                matched_idx = choices.index(answer)
            elif isinstance(answer, str):
                a_low = answer.lower().strip()
                a_words = set(re.findall(r"\w+", a_low))

                for i, choice in enumerate(choices):
                    if not isinstance(choice, str):
                        continue
                    c_low = choice.lower().strip()
                    c_words = set(re.findall(r"\w+", c_low))

                    if a_low in c_low or c_low in a_low:
                        matched_idx = i
                        break
                    if (
                        a_words
                        and c_words
                        and (a_words <= c_words or c_words <= a_words)
                    ):
                        matched_idx = i
                        break

        if matched_idx == -1:
            etc_count += 1
            print(f"idx: {idx}")
            print(f"answer: {repr(answer)}")
            print(f"choices: {choices}")
            print()

    print(f"Total etc: {etc_count}")


def output_csv(category: str, counter: dict, total: int, output_path: str) -> None:
    """CSV 출력"""
    rows = []
    sorted_choices = sorted(counter.keys(), key=lambda x: (x == "etc", x))
    for choice in sorted_choices:
        count = counter[choice]
        pct = count / total * 100 if total > 0 else 0
        rows.append(
            {
                "category": category,
                "choice": choice,
                "count": count,
                "percentage": round(pct, 2),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="model_response와 실제 정답(answer)의 선택지 분포를 함께 출력합니다."
    )
    parser.add_argument(
        "--parquet-dir",
        default="~/.cache/huggingface/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots/*/",
        help="parquet 파일들이 있는 폴더 또는 단일 파일 경로",
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
    parquet_dir = Path(args.parquet_dir).expanduser()

    try:
        # model_response 분포
        model_counter, total = count_choices(
            str(parquet_dir),
            args.category,
            args.response_col,
            include_ground_truth=False,
        )
        # ground truth 분포
        gt_counter, total = count_choices(
            str(parquet_dir),
            args.category,
            args.response_col,
            include_ground_truth=True,
        )

        if args.output_mode == "cli":
            output_cli(f"Model Response - {args.category}", model_counter, total)
            print()
            output_cli(f"Ground Truth - {args.category}", gt_counter, total)
            print()
            print("=== ETC Details (정답이 choices에 없는 경우) ===")
            output_etc_details(str(parquet_dir), args.category, args.response_col)
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
