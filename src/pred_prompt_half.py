import argparse
import glob
import os
import warnings

import librosa
import pandas as pd
from sacrebleu.metrics import BLEU
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


def make_prompt(category: str, question: str, choices: list[str]) -> str:
    prompt = []
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
    bleu_metric = BLEU()

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        prompts = []
        audios = []
        ground_truths = []

        # 2. 배치 내 데이터 준비
        for _, row in batch_df.iterrows():
            # 프롬프트 생성
            full_prompt = make_prompt(row["category"], row["question"], row["choices"])

            # 프롬프트를 단어 단위로 절반으로 자르기
            words = full_prompt.split()
            half_idx = len(words) // 2
            first_half = " ".join(words[:half_idx])
            second_half = " ".join(words[half_idx:])

            prompts.append(first_half)
            ground_truths.append(second_half)

            if not use_noise:
                audio_path = os.path.join(data_dir, row["audio_path"][0])
            else:
                audio_path = os.path.join("./assets", "white-noise.mp3")
            audios.append(librosa.load(audio_path, sr=None)[0])

        # 3. 배치 추론 실행
        print(f"Processing batch: {i // batch_size + 1} (size: {len(prompts)})")
        batch_responses = model.inference(prompts, audios)

        # 4. 결과 수집 및 평가
        all_responses.extend(batch_responses)

        # (선택) 중간 결과 출력 및 평가 메트릭 계산
        for idx, res in enumerate(batch_responses):
            gt = ground_truths[idx]

            # EM (Exact Match) 계산
            em_score = 1.0 if res.strip() == gt.strip() else 0.0

            # BLEU 계산
            bleu_score = bleu_metric.sentence_score(res, [gt]).score

            print(f"Sample {i + idx}:")
            print("-" * 50)
            print(f"Input (first half): \n{prompts[idx]}")
            print("-" * 50)
            print(f"Ground Truth (second half): \n{gt}")
            print("-" * 50)
            print(f"Model Response: \n{res}")
            print("-" * 50)
            print(f"EM: {em_score}, BLEU: {bleu_score:.2f}")
            print("-" * 50)

            df.at[i + idx, "input_prompt"] = prompts[idx]
            df.at[i + idx, "ground_truth"] = gt
            df.at[i + idx, "model_response"] = res
            df.at[i + idx, "em_score"] = em_score
            df.at[i + idx, "bleu_score"] = bleu_score

        os.makedirs("./results", exist_ok=True)
        df.to_csv(
            os.path.join(
                "./results",
                f"{model_name.replace('.', '')}_pred_half_{'noise' if use_noise else 'audio'}.csv",
            )
        )
    # 최종 통계 계산 및 출력
    avg_em = df["em_score"].mean()
    avg_bleu = df["bleu_score"].mean()

    print(f"\n{'=' * 50}")
    print(f"Total processed: {len(all_responses)}")
    print(f"Average EM: {avg_em:.4f} ({avg_em * 100:.2f}%)")
    print(f"Average BLEU: {avg_bleu:.2f}")
    print(f"{'=' * 50}")

    os.makedirs("./results", exist_ok=True)
    # 통계 요약 저장
    summary = {
        "model": model_name,
        "use_noise": use_noise,
        "total_samples": len(all_responses),
        "avg_em": avg_em,
        "avg_bleu": avg_bleu,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        os.path.join(
            "./results",
            f"{model_name.replace('.', '')}_summary_half_{'noise' if use_noise else 'audio'}.csv",
        ),
        index=False,
    )
