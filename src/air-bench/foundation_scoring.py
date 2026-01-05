import argparse
import json
from collections import defaultdict
from pathlib import Path

CATEGORY_LABELS: tuple[tuple[str, str], ...] = (
    ("Speech_Grounding", "Speech grounding"),
    ("Spoken_Language_Identification", "Spoken language identification"),
    ("Speaker_Gender_Recognition", "Speaker gender recognition"),
    ("Speaker_Emotion_Recontion", "Emotion recognition"),
    ("Speaker_Age_Prediction", "Speaker age prediction"),
    ("Speech_Entity_Reconition", "Speech entity recognition"),
    ("Speaker_Intent_Classification", "Intent classification"),
    ("Speaker_Number_Verification", "Speaker number verification"),
    ("Synthesized_Voice_Detection", "Synthesized voice detection"),
    ("Audio_Grounding", "Audio grounding"),
    ("vocal_sound_classification", "Vocal sound classification"),
    ("Acoustic_Scene_Classification", "Acoustic scene classification"),
    ("Sound_AQA", "Sound question answering"),
    ("Music_Instruments_Classfication", "Music instruments classification"),
    ("Music_Genre_Recognition", "Music genre classification"),
    ("Music_Midi_Pitch_Analysis", "Music note analysis-pitch"),
    ("Music_Midi_Velocity_Analysis", "Music note analysis-velocity"),
    ("Music_AQA", "Music question answering"),
    ("Music_Mood_Recognition", "Music emotion detection"),
)

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
        help=(
            "Input jsonl path from foundation_infer.py (default output: "
            "./results/air-bench/{model}_predictions_foundation.jsonl)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output summary json path (defaults to input name + _foundation_scores.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_stem = input_path.stem.replace("predictions", "evaluation", 1)
        output_path = input_path.with_name(f"{output_stem}.json")

    fail_num = 0
    fail_response_parse_total = 0
    task_id_list: list[str] = []
    total_num_dict: dict[str, int] = defaultdict(int)
    total_all_num_dict: dict[str, int] = defaultdict(int)
    parse_fail_num_dict: dict[str, int] = defaultdict(int)
    correct_num_dict: dict[str, int] = defaultdict(int)
    category_total: dict[str, int] = defaultdict(int)
    category_total_all: dict[str, int] = defaultdict(int)
    category_parse_fail: dict[str, int] = defaultdict(int)
    category_correct: dict[str, int] = defaultdict(int)

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

            total_all_num_dict[task_id] += 1
            category_total_all[task_name] += 1

            response = record.get("response")
            pred = _parse_prediction_letter(response)
            if pred is None:
                print(
                    f"[{line_no}] response parse failed: task_id={task_id} uniq_id={record.get('uniq_id')} response={repr(response)[:100]}"
                )
                fail_response_parse_total += 1
                fail_num += 1
                parse_fail_num_dict[task_id] += 1
                category_parse_fail[task_name] += 1
                continue

            gt = _ground_truth_letter(record)

            total_num_dict[task_id] += 1
            if gt == pred:
                correct_num_dict[task_id] += 1
            category_total[task_name] += 1
            if gt == pred:
                category_correct[task_name] += 1

    total_sum = 0
    total_sum_all = 0
    total_correct = 0
    for task_id in task_id_list:
        total_num = total_num_dict[task_id]
        total_all_num = total_all_num_dict[task_id]
        parse_fail_num = parse_fail_num_dict[task_id]
        correct_num = correct_num_dict[task_id]
        acc = correct_num / total_num if total_num else 0.0
        acc_all = correct_num / total_all_num if total_all_num else 0.0
        total_sum += total_num
        total_sum_all += total_all_num
        total_correct += correct_num
        print(
            f"{task_id}: Sum={total_num}, correct={correct_num}, acc={acc}, acc_all={acc_all}, parse_fail={parse_fail_num}"
        )

    category_scores: dict[str, dict[str, float | int | str]] = {}
    category_accs: list[float] = []
    print("=" * 100)
    print("Category scores")
    label_width = max(len(label) for _, label in CATEGORY_LABELS)
    for task_name, label in CATEGORY_LABELS:
        total_num = category_total.get(task_name, 0)
        total_all_num = category_total_all.get(task_name, 0)
        parse_fail_num = category_parse_fail.get(task_name, 0)
        correct_num = category_correct.get(task_name, 0)
        acc = correct_num / total_num if total_num else 0.0
        acc_all = correct_num / total_all_num if total_all_num else 0.0
        category_scores[label] = {
            "task_name": task_name,
            "total": total_num,
            "total_all": total_all_num,
            "parse_fail": parse_fail_num,
            "correct": correct_num,
            "acc": acc,
            "acc_all": acc_all,
        }
        if total_num:
            category_accs.append(acc)
        print(
            f"{label:<{label_width}} | Sum={total_num:5d} | correct={correct_num:5d} | acc={acc:7.2%} | acc_all={acc_all:7.2%} | parse_fail={parse_fail_num:5d}"
        )

    category_average = (
        (sum(category_accs) / len(category_accs)) * 100.0 if category_accs else 0.0
    )

    overall_acc_all = (total_correct / total_sum_all) if total_sum_all else 0.0

    print("-" * 100)
    print(f"Total sum: {total_sum}")
    print(f"Total correct: {total_correct}")
    print(f"fail_num: {fail_num}")
    print(f"fail_response_parse: {fail_response_parse_total}")
    print(f"acc_all(overall): {overall_acc_all:.4f}")
    print(f"category_average: {category_average:.2f}%")

    summary = {
        "input": str(input_path),
        "tasks": {
            task_id: {
                "total": total_num_dict[task_id],
                "total_all": total_all_num_dict[task_id],
                "parse_fail": parse_fail_num_dict[task_id],
                "correct": correct_num_dict[task_id],
                "acc": correct_num_dict[task_id] / total_num_dict[task_id]
                if total_num_dict[task_id]
                else 0.0,
                "acc_all": correct_num_dict[task_id] / total_all_num_dict[task_id]
                if total_all_num_dict[task_id]
                else 0.0,
            }
            for task_id in task_id_list
        },
        "categories": category_scores,
        "category_average": category_average,
        "total_sum": total_sum,
        "total_sum_all": total_sum_all,
        "total_correct": total_correct,
        "fail_num": fail_num,
        "fail_response_parse": fail_response_parse_total,
        "acc_all": overall_acc_all,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(f"summary saved: {output_path}")


if __name__ == "__main__":
    main()
