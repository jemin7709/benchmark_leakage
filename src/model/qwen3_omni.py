import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import warnings

import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    AutoProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm import LLM, SamplingParams
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Qwen3Omni_HF:
    def __init__(self, model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"):
        self.model_name = model_name
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.system_prompt = ""

    def inference(self, prompts: list[str], audio_paths: list[object]) -> list[str]:
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            for prompt, audio_path in zip(prompts, audio_paths)
        ]

        inputs_text = [
            self.processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )
            for message in messages
        ]

        audios_list = []
        for message in messages:
            audios, _, _ = process_mm_info(message, use_audio_in_video=True)
            audios_list.extend(audios)

        inputs = self.processor(
            text=inputs_text,
            audio=audios_list,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, self.model.dtype)

        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=4096,
            do_sample=False,
        )

        responses = [
            self.processor.decode(
                output[input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for output in outputs
        ]

        return responses


class Qwen3Omni_VLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Thinking"):
        self.model_name = model_name
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            max_model_len=65536,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.system_prompt = ""
        self.reasoning_parser = Qwen3ReasoningParser(self.processor.tokenizer)

    def inference(self, prompts: list[str], audio_paths: list[object]):
        sampling_params = SamplingParams(
            max_tokens=4096, temperature=0.0, truncate_prompt_tokens=-1
        )

        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            for prompt, audio_path in zip(prompts, audio_paths)
        ]

        inputs_text = [
            self.processor.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )
            for message in messages
        ]
        audios_list: list[object] = []
        for message in messages:
            audios, _, _ = process_mm_info(message, use_audio_in_video=True)
            audios_list.append(audios)
        inputs = [
            {
                "prompt": inputs_text,
                "multi_modal_data": {"audio": audios} if audios is not None else {},
            }
            for inputs_text, audios in zip(inputs_text, audios_list)
        ]

        outputs = self.model.generate(inputs, sampling_params=sampling_params)

        return [
            self.reasoning_parser.extract_reasoning(output.outputs[0].text, None)[1]
            for output in outputs
        ]


if __name__ == "__main__":
    from transformers import set_seed
    from vllm.assets.audio import AudioAsset

    set_seed(0)

    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate[0]
    model = Qwen3Omni_HF()
    responses = model.inference(
        ["Transcribe this audio into English, and then translate it into French."] * 3,
        [audio] * 3,
    )

    print("=" * 100)
    for i, response in enumerate(responses):
        print(f"[{i}] {response}")
        print("-" * 100)
