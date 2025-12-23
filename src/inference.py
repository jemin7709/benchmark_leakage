import argparse
import glob
import os
import warnings

import librosa
import pandas as pd
from transformers import set_seed

from model.gemma3n import Gemma3n_VLLM
from model.qwen3_omni import Qwen3Omni_VLLM
from model.qwen25_omni import Qwen2_5Omni_VLLM

warnings.filterwarnings("ignore", category=UserWarning)


def load_data() -> tuple[pd.DataFrame, str]:
    data_root = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots"
    )
    parquet_dirs = glob.glob(os.path.join(data_root, "*", "test.parquet"))
    return pd.read_parquet(parquet_dirs[0]), os.path.dirname(parquet_dirs[0])


def make_prompt(
    category: str, question: str, choices: list[str], transcription: str
) -> str:
    prompt = []
    if category.lower() == "instruction following":
        prompt.append("Audio Transcript: " + transcription + "\n")
    prompt.append("Question: " + question + "\n")
    if category.lower() != "open" and category.lower() != "instruction following":
        prompt.append("Options:\n")
        choices_list = [
            f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)
        ]
        prompt.append("\n".join(choices_list))
    return "".join(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3n",
        choices=["gemma3n", "qwen3-omni", "qwen2.5-omni"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--use-noise",
        action="store_true",
        help="Use white noise audio instead of actual audio files",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    args = parser.parse_args()
    set_seed(0)

    batch_size: int = args.batch_size
    model_name = args.model
    use_noise = args.use_noise

    df, data_dir = load_data()
    if model_name == "gemma3n":
        model = Gemma3n_VLLM()
    elif model_name == "qwen3-omni":
        model = Qwen3Omni_VLLM()
    elif model_name == "qwen2.5-omni":
        model = Qwen2_5Omni_VLLM()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    all_responses = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        prompts = []
        audios = []

        # 2. 배치 내 데이터 준비
        for _, row in batch_df.iterrows():
            # 프롬프트 생성
            prompt = make_prompt(
                row["category"], row["question"], row["choices"], row["transcription"]
            )
            prompts.append(prompt)

            if not use_noise:
                audio_path = os.path.join(data_dir, row["audio_path"][0])
            else:
                audio_path = os.path.join("./assets", "white-noise.mp3")
            audios.append(librosa.load(audio_path, sr=None)[0])

        # 3. 배치 추론 실행
        print(f"Processing batch: {i // batch_size + 1} (size: {len(prompts)})")
        batch_responses = model.inference(prompts, audios)

        # 4. 결과 수집
        all_responses.extend(batch_responses)

        # (선택) 중간 결과 출력
        for idx, res in enumerate(batch_responses):
            print(f"Sample {i + idx}: \n{prompts[idx]}\n\n{res}")
            print("-" * 50)
            df.at[i + idx, "model_response"] = res

        os.makedirs("./results", exist_ok=True)
        df.to_parquet(
            os.path.join(
                "./results", f"{model_name.replace('.', '')}_predictions_{"noise" if use_noise else "audio"}.parquet"
            )
        )
    print(f"Total processed: {len(all_responses)}")
