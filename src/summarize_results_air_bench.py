import argparse
import glob
import json
import os

import pandas as pd


def format_percentage(val):
    if val is None:
        return "-"
    try:
        return f"{float(val) * 100:.2f}"
    except (ValueError, TypeError):
        return "-"


def get_row_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    categories_to_extract = [
        "Speech_Grounding",
        "Spoken_Language_Identification",
        "Speaker_Gender_Recognition",
        "Speaker_Age_Prediction",
        "Speech_Entity_Reconition",
        "Speaker_Intent_Classification",
        "Speaker_Number_Verification",
        "Synthesized_Voice_Detection",
        "Audio_Grounding",
        "vocal_sound_classification",
        "Acoustic_Scene_Classification",
        "Sound_AQA",
        "Music_Instruments_Classfication",
        "Music_Genre_Recognition",
        "Music_Midi_Pitch_Analysis",
        "Music_Midi_Velocity_Analysis",
        "Music_AQA",
        "Music_Mood_Recognition",
    ]

    model_name = (
        os.path.basename(file_path)
        .replace("_evaluation_foundation", "")
        .replace(".json", "")
    )

    row = {"Model": model_name}
    cat_results = data.get("categories", {})

    task_name_to_data = {}
    for cat_data in cat_results.values():
        t_name = cat_data.get("task_name")
        if t_name:
            task_name_to_data[t_name] = cat_data

    for task_name in categories_to_extract:
        score = None
        if task_name in task_name_to_data:
            score = task_name_to_data[task_name].get("acc_all")
        row[task_name] = format_percentage(score)

    avg_score = data.get("category_average")
    if avg_score is not None:
        row["Average"] = f"{float(avg_score):.2f}"
    else:
        row["Average"] = "-"

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        help="JSON files to summarize. If empty, searches results/air-bench/",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output in CSV format instead of table",
    )

    args = parser.parse_args()

    if not args.files:
        results_dir = "results/air-bench"
        files = glob.glob(os.path.join(results_dir, "*_evaluation_foundation*.json"))
        files.sort()
    else:
        files = args.files

    if not files:
        print("No evaluation results found.")
        return

    summary_data = []
    for f in files:
        if os.path.exists(f):
            summary_data.append(get_row_data(f))
        else:
            print(f"Warning: File not found: {f}")

    if not summary_data:
        return

    df = pd.DataFrame(summary_data)

    categories_to_extract = [
        "Speech_Grounding",
        "Spoken_Language_Identification",
        "Speaker_Gender_Recognition",
        "Speaker_Age_Prediction",
        "Speech_Entity_Reconition",
        "Speaker_Intent_Classification",
        "Speaker_Number_Verification",
        "Synthesized_Voice_Detection",
        "Audio_Grounding",
        "vocal_sound_classification",
        "Acoustic_Scene_Classification",
        "Sound_AQA",
        "Music_Instruments_Classfication",
        "Music_Genre_Recognition",
        "Music_Midi_Pitch_Analysis",
        "Music_Midi_Velocity_Analysis",
        "Music_AQA",
        "Music_Mood_Recognition",
        "Average",
    ]

    cols = ["Model"] + categories_to_extract
    cols = [c for c in cols if c in df.columns]

    df = df[cols]

    if args.csv:
        print(df.to_csv(index=False))
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 5000)
        pd.set_option("display.expand_frame_repr", False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
