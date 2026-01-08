#!/usr/bin/env python3
import pathlib
import subprocess

from model_config import get_model_preset
from persona_config import get_persona_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-run"
PRESET = get_model_preset()
PERSONA = get_persona_preset()
MODEL = PRESET.model_gguf
PREFIX = PERSONA.ttsv_prefix(PRESET.key)
CHAT_TEMPLATE = PRESET.chat_template

scales = ["1", "0.3", "0.1", "0.03"]
MAX_TOKENS = f"{PERSONA.max_tokens_eval}"
TEMP = f"{PERSONA.eval_temp}"
TOP_K = f"{PERSONA.eval_top_k}"
TOP_P = f"{PERSONA.eval_top_p}"
REPEAT_PENALTY = f"{PERSONA.eval_repeat_penalty}"

prompts = PERSONA.sweep_prompts or PERSONA.eval_prompts

base_args = [
    str(LLAMA),
    "-m",
    str(MODEL),
    "-c",
    "512",
    "-n",
    MAX_TOKENS,
    "--temp",
    TEMP,
    "--top-k",
    TOP_K,
    "--top-p",
    TOP_P,
    "--repeat-penalty",
    REPEAT_PENALTY,
    "--system-prompt",
    PERSONA.system_prompt,
    "--chat-template-file",
    str(CHAT_TEMPLATE),
    "--jinja",
    "--log-disable",
]

for prompt in prompts:
    print("=" * 80)
    print("PROMPT:", prompt)

    cmd_base = base_args + ["-p", prompt]
    base_proc = subprocess.run(
        cmd_base, text=True, errors="replace", capture_output=True
    )
    if base_proc.returncode != 0:
        raise RuntimeError(base_proc.stderr)
    base_out = base_proc.stdout
    print("\nBASELINE:\n", base_out.strip())

    for scale in scales:
        cmd_ttsv = base_args + [
            "-p",
            prompt,
            "--ttsv",
            str(PREFIX),
            "--ttsv-scale",
            scale,
        ]
        ttsv_proc = subprocess.run(
            cmd_ttsv, text=True, errors="replace", capture_output=True
        )
        if ttsv_proc.returncode != 0:
            raise RuntimeError(ttsv_proc.stderr)
        ttsv_out = ttsv_proc.stdout
        print(f"\nTTSV scale={scale}:\n", ttsv_out.strip())
