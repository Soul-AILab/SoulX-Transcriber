import os
import re
import json
import numpy as np
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

from utils import detect_and_fix_hallucination_repetition, parse_time

SEED = 42

sdr_system = (
    "You are an expert in speaker diarization and automatic speech recognition."
)

# user prompt without punctuation
USER_QUERY = """
    Task: Speaker Diarization and ASR. 
    Rules:
    1. Identify each speaker and their spoken content without punctuation.
    2. Format each turn as: [start_time --> end_time] Speaker X: text    
    3. Timestamps should be precise to the millisecond (e.g., 00:00:01.234).
    4. Do NOT split an utterance to avoid overlap — keep each speaker turn complete.
    5. Handle overlapping speech by showing concurrent turns with their own time ranges.
    6. Output only the formatted results. No preamble, no explanation.
"""

# USER_QUERY = """
# Task: Speaker Diarization and ASR. 
#     Rules:
#     1. Identify each speaker and their spoken content with punctuation.
#     2. Format each turn as: [start_time --> end_time] Speaker X: text    
#     3. Timestamps should be precise to the millisecond (e.g., 00:00:01.234).
#     4. Do NOT split an utterance to avoid overlap — keep each speaker turn complete.
#     5. Handle overlapping speech by showing concurrent turns with their own time ranges.
#     6. Output only the formatted results. No preamble, no explanation.
# <audio>
# """


def validate_format(hyp):
    total_len = 0
    validated_len = 0
    pattern = re.compile(r'\[(\d{2}:\d{2}\.\d{2,3})\s*-->\s*(\d{2}:\d{2}\.\d{2,3})\]\s*(Speaker \d+):\s*(.*)')
    for line in hyp.strip().split('\n'):
        match = pattern.match(line.strip())
        if match:
            start_time_str, end_time_str, speaker, text = match.groups()
            start_time = parse_time(start_time_str)
            end_time = parse_time(end_time_str)
            if end_time > start_time:
                validated_len += 1
            total_len += 1

    if total_len == 0:
        print(f"format validation failed: no matching lines")
        return False
    elif validated_len / total_len < 0.8:
        print(f"format validation failed: {validated_len}/{total_len} < 80%")
        return False
    return True


def get_audio_query(audio_path: str, sampling_rate: int = 16000) -> dict:
    """single audio inference input"""
    prompt = (
        f"<|im_start|>system\n{sdr_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{USER_QUERY}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
    audio_data = (audio_signal.astype(np.float32), sr)

    return {
        "prompt": prompt,
        "multi_modal_data": {"audio": audio_data},
    }


def init_model(args):
    omni = Omni(
        model=args.model,
        dtype=args.dtype,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.init_timeout,
    )

    thinker_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
        logit_bias={},
        seed=SEED,
    )

    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[2150],
    )

    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096 * 16,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    all_sampling_params = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]
    num_stages = omni.num_stages
    sampling_params_list = all_sampling_params[:num_stages]

    return omni, sampling_params_list


def decode_single(args):
    """single audio inference and write results"""

    # ── initialize model ──
    omni, sampling_params_list = init_model(args)

    # retry loop for single inference to get valid format
    for attempt in range(args.MAX_RETRIES):
        # ── output file name ──
        wav_name = os.path.splitext(os.path.basename(args.audio_path))[0]

        # ── build input ──
        query_input = get_audio_query(args.audio_path, args.sampling_rate)
        prompts = [query_input for _ in range(args.num_prompts)]

        # ── output directory ──
        os.makedirs(args.output_dir, exist_ok=True)
        jsonl_path = os.path.join(args.output_dir, f"{wav_name}.jsonl")

        # ── inference ──
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:

            omni_generator = omni.generate(
                prompts, sampling_params_list, py_generator=args.py_generator
            )
            for stage_outputs in omni_generator:
                output = stage_outputs.request_output
                if stage_outputs.final_output_type == "text":
                    text_output = output.outputs[0].text
                    is_valid = validate_format(text_output)
                    if is_valid:
                        print(f"[SUCCESS] get the valid format in {attempt + 1} attempts.")
                        text_output = detect_and_fix_hallucination_repetition(text_output)
                        has_hallucination = text_output["has_hallucination"]
                        if has_hallucination:
                            print(f"[WARNING] Detected hallucination in the output, please check the result carefully.")
                        final_text = text_output["repaired_text"] if has_hallucination else text_output["original_text"]
                        jsonl_f.write(
                            json.dumps(
                                {"index": wav_name, "hyp": final_text},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        jsonl_f.flush()
                        print(f"\n[OUTPUT]\n{final_text}")
                        return
                    else:
                        print(f"[WARNING] Attempt {attempt + 1} failed format validation, retrying...")
                        args.temperature = min(args.temperature + 0.2 * (attempt + 1), 1.0)
                    break

    print(f"\n[DONE] Results saved to {args.output_dir}/")
    print(f"  JSONL: {jsonl_path}")

    omni.close()


def parse_args():
    parser = FlexibleArgumentParser(
        description="Single WAV inference with Qwen3-Omni (speaker diarization + ASR)"
    )
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--audio-path", "-a", type=str, required=True,
                        help="Path to the input WAV file.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save STM / text / JSONL outputs.")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--stage-configs-path", type=str, default=None)
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--MAX-RETRIES", type=int, default=3)
    parser.add_argument("--stage-init-timeout", type=int, default=6000)
    parser.add_argument("--init-timeout", type=int, default=6000)
    parser.add_argument("--py-generator", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    decode_single(args)
