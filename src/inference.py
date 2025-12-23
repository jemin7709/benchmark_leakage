import glob
import os

import librosa
import pandas as pd
from transformers import set_seed

from model.qwen25_omni import Qwen2_5Omni_HF

set_seed(0)
# 1. 모델 로드
# 2. 오디오 및 프롬프트 준비
# 3. 추론 실행

limit: int | None = None
batch_size: int = 8


def load_data() -> tuple[pd.DataFrame, str]:
    data_root = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots"
    )
    parquet_dirs = glob.glob(os.path.join(data_root, "*", "test.parquet"))
    return pd.read_parquet(parquet_dirs[0]), os.path.dirname(parquet_dirs[0])


def make_prompt(
    category: str, question: str, choices: list[str], transcription: str
) -> str:
    prompt = [
        "Answer the following multiple choice question.\n\n",
        question,
        "\n\n",
    ]
    if category.lower() != "open":
        if category.lower() == "instruction following":
            prompt.append(transcription)
        else:
            prompt.append("Options:\n")
            choices_list = [
                f"{chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)
            ]
            prompt.append("\n".join(choices_list))
    return "".join(prompt)


df, data_dir = load_data()
model = Qwen2_5Omni_HF()

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

        audio_path = os.path.join(data_dir, row["audio_path"][0])
        audios.append(librosa.load(audio_path, sr=None)[0])

    # 3. 배치 추론 실행
    print(f"Processing batch: {i // batch_size + 1} (size: {len(prompts)})")
    batch_responses = model.inference(prompts, audios)

    # 4. 결과 수집
    all_responses.extend(batch_responses)

    # (선택) 중간 결과 출력
    for idx, res in enumerate(batch_responses):
        print(f"Sample {i + idx}: {res}")

print(f"Total processed: {len(all_responses)}")
