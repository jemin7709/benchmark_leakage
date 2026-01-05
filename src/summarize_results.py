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

    mapping = {
        "Sound": ["sound"],
        "Music": ["music"],
        "Speech": ["speech"],
        "Sound-Music": ["sound_music"],
        "Speech-Music": ["music_speech"],
        "Speech-Sound": ["sound_speech"],
        "Sound-Music-Speech": ["sound_music_speech"],
        "Spatial": ["spatial_audio"],
        "Voice": ["voice_chat"],
        "Multi-Audio": ["multi"],
        "Open-ended": ["open"],
        "IF": ["instruction following"],
    }

    model_name = os.path.basename(file_path).replace("_comprehensive_results.json", "")

    row = {"Model": model_name}
    cat_results = data.get("category_results", {})

    for label, keys in mapping.items():
        score = None
        for key in keys:
            if key in cat_results:
                score = cat_results[key].get("performance_score")
                break
        row[label] = format_percentage(score)

    avg_score = data.get("evaluation_summary", {}).get("overall_weighted_performance")
    row["Avg"] = format_percentage(avg_score)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        help="JSON files to summarize. If empty, searches results/mmau-pro/",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV format",
    )

    args = parser.parse_args()

    if not args.files:
        results_dir = "results/mmau-pro"
        files = glob.glob(os.path.join(results_dir, "*_comprehensive_results.json"))
        files.sort()
    else:
        files = args.files

    if not files:
        print("No comprehensive results found.")
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

    labels = [
        "Sound",
        "Music",
        "Speech",
        "Sound-Music",
        "Speech-Music",
        "Speech-Sound",
        "Sound-Music-Speech",
        "Spatial",
        "Voice",
        "Multi-Audio",
        "Open-ended",
        "IF",
        "Avg",
    ]

    cols = ["Model"] + labels
    cols = [c for c in cols if c in df.columns]

    df = df[cols]

    if args.csv:
        print(df.to_csv(index=False))
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 2000)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
