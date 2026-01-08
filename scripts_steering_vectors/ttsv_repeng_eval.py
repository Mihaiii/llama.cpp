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
CVEC = PERSONA.repeng_vector(PRESET.key)
CHAT_TEMPLATE = PRESET.chat_template

TTSV_SCALE = f"{PERSONA.ttsv_scale}"
TTSV_BLEND = f"{PERSONA.ttsv_blend}"
TTSV_BLEND_SCHEDULE = PERSONA.ttsv_blend_schedule
CVEC_SCALE = "0.2"
CVEC_LAYER_RANGE = ("6", "10")  # middle layers
COLLAPSE_WINDOW = "4"
COLLAPSE_PATIENCE = "1"
MAX_TOKENS = f"{PERSONA.max_tokens_repeng_eval}"
TEMP = f"{PERSONA.eval_temp}"
TOP_K = f"{PERSONA.eval_top_k}"
TOP_P = f"{PERSONA.eval_top_p}"
REPEAT_PENALTY = f"{PERSONA.eval_repeat_penalty}"

prompts = PERSONA.eval_prompts

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
    "--ttsv-collapse-window",
    COLLAPSE_WINDOW,
    "--ttsv-collapse-patience",
    COLLAPSE_PATIENCE,
    "--system-prompt",
    PERSONA.system_prompt,
    "--chat-template-file",
    str(CHAT_TEMPLATE),
    "--jinja",
    "--log-disable",
]


def run(cmd):
    proc = subprocess.run(cmd, text=True, errors="replace", capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()


def add_cvec_args(args):
    out = list(args)
    out += ["--control-vector-scaled", f"{CVEC}:{CVEC_SCALE}"]
    if CVEC_LAYER_RANGE:
        out += [
            "--control-vector-layer-range",
            CVEC_LAYER_RANGE[0],
            CVEC_LAYER_RANGE[1],
        ]
    return out


for prompt in prompts:
    print("=" * 80)
    print("PROMPT:", prompt)

    base_out = run(base_args + ["-p", prompt])
    print("\nBASELINE:\n", base_out)

    ttsv_out = run(
        base_args
        + [
            "-p",
            prompt,
            "--ttsv",
            str(PREFIX),
            "--ttsv-scale",
            TTSV_SCALE,
            "--ttsv-logit-blend",
            TTSV_BLEND,
            "--ttsv-logit-blend-schedule",
            TTSV_BLEND_SCHEDULE,
        ]
    )
    print(f"\nTTSV scale={TTSV_SCALE}:\n", ttsv_out)

    repeng_out = run(
        add_cvec_args(
            base_args
            + [
                "-p",
                prompt,
                "--ttsv-logit-blend",
                TTSV_BLEND,
                "--ttsv-logit-blend-schedule",
                TTSV_BLEND_SCHEDULE,
            ]
        )
    )
    print(f"\nREPENG scale={CVEC_SCALE}:\n", repeng_out)

    both_out = run(
        add_cvec_args(
            base_args
            + [
                "-p",
                prompt,
                "--ttsv",
                str(PREFIX),
                "--ttsv-scale",
                TTSV_SCALE,
                "--ttsv-logit-blend",
                TTSV_BLEND,
                "--ttsv-logit-blend-schedule",
                TTSV_BLEND_SCHEDULE,
            ]
        )
    )
    print(f"\nTTSV+REPENG (ttsv={TTSV_SCALE}, cvec={CVEC_SCALE}):\n", both_out)
