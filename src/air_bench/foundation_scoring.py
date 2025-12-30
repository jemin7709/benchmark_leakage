import argparse
import json
from collections import defaultdict
from pathlib import Path


def _parse_prediction_letter(response: object) -> str | None:
    if response is None:
        return None

    if not isinstance(response, str):
        response = str(response)

    predict = response.strip().replace("\n", "")
    if not predict or predict == "None":
        return None

    first = predict[0]
    if first in {"A", "B", "C", "D"}:
        return first

    if len(predict) > 1:
        maybe = predict[-2]
        if maybe in {"A", "B", "C", "D"}:
            return maybe

    return None


def _ground_truth_letter(record: dict) -> str:
    answer_gt = record.get("answer_gt")

    if answer_gt == record.get("choice_a"):
        return "A"
    if answer_gt == record.get("choice_b"):
        return "B"
    if answer_gt == record.get("choice_c", None):
        return "C"
    if answer_gt == record.get("choice_d", None):
        return "D"

    raise ValueError(f"Could not map answer_gt to A/B/C/D: answer_gt={answer_gt!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input jsonl path from foundation_infer.py",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    fail_num = 0
    task_id_list: list[str] = []
    total_num_dict: dict[str, int] = defaultdict(int)
    correct_num_dict: dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            raw = raw.strip()
            if not raw:
                continue

            record = json.loads(raw)

            task_name = record.get("task_name")
            dataset_name = record.get("dataset_name")
            if task_name is None or dataset_name is None:
                print(f"[{line_no}] task_name/dataset_name is None")
                fail_num += 1
                continue

            task_id = f"{task_name}_{dataset_name}"
            if task_id not in task_id_list:
                task_id_list.append(task_id)

            response = record.get("response")
            pred = _parse_prediction_letter(response)
            if pred is None:
                print(
                    f"[{line_no}] response parse failed: task_id={task_id} uniq_id={record.get('uniq_id')} response={repr(response)[:100]}"
                )
                fail_num += 1
                continue

            gt = _ground_truth_letter(record)

            total_num_dict[task_id] += 1
            if gt == pred:
                correct_num_dict[task_id] += 1

    total_sum = 0
    total_correct = 0
    for task_id in task_id_list:
        total_num = total_num_dict[task_id]
        correct_num = correct_num_dict[task_id]
        acc = correct_num / total_num if total_num else 0.0
        total_sum += total_num
        total_correct += correct_num
        print(f"{task_id}: Sum={total_num}, correct={correct_num}, acc={acc}")

    print(f"total_sum: {total_sum}")
    print(f"total_correct: {total_correct}")
    print(f"fail_num: {fail_num}")


if __name__ == "__main__":
    main()
