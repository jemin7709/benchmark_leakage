import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import warnings

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Gemma3n_HF:
    """Gemma 3n multimodal model wrapper for HuggingFace Transformers.

    Supports audio-to-text inference using Gemma3nForConditionalGeneration.
    """

    def __init__(self, model_name: str = "google/gemma-3n-E4B-it"):
        self.model_name = model_name
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.system_prompt = ""

    def inference(self, prompts: list[str], audio_paths: list[object]) -> list[str]:
        """Run inference on audio inputs.

        Args:
            prompts: List of text prompts.
            audio_paths: List of audio inputs (file paths or numpy arrays).

        Returns:
            List of generated text responses.
        """
        messages_batch = [
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

        responses = []
        for messages in messages_batch:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)

            input_len = inputs["input_ids"].shape[-1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                disable_compile=True,
            )

            text = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            responses.append(text[0])

        return responses


class Gemma3n_VLLM:
    """Gemma 3n multimodal model wrapper for vLLM.

    Supports audio-to-text inference using vLLM engine.
    """

    def __init__(self, model_name: str = "google/gemma-3n-E4B-it"):
        self.model_name = model_name
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.system_prompt = ""

    def inference(self, prompts: list[str], audio_paths: list[object]) -> list[str]:
        """Run inference on audio inputs.

        Args:
            prompts: List of text prompts.
            audio_paths: List of audio inputs (file paths or numpy arrays).

        Returns:
            List of generated text responses.
        """
        sampling_params = SamplingParams(
            max_tokens=4096, temperature=0.0, truncate_prompt_tokens=-1
        )

        messages_batch = [
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
                messages, add_generation_prompt=True, tokenize=False
            )
            for messages in messages_batch
        ]

        inputs = [
            {
                "prompt": text,
                "multi_modal_data": {"audio": audio},
            }
            for text, audio in zip(inputs_text, audio_paths)
        ]

        outputs = self.model.generate(inputs, sampling_params=sampling_params)

        return [output.outputs[0].text for output in outputs]


if __name__ == "__main__":
    from transformers import set_seed
    from vllm.assets.audio import AudioAsset

    set_seed(0)

    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate[0]
    model = Gemma3n_VLLM()
    responses = model.inference(
        ["Describe this audio."] * 2,
        [audio] * 2,
    )

    print("=" * 100)
    for i, response in enumerate(responses):
        print(f"[{i}] {response}")
        print("-" * 100)
