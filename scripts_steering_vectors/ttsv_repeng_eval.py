#!/usr/bin/env python3
import pathlib
import subprocess

from model_config import get_model_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-run"
PRESET = get_model_preset()
MODEL = PRESET.model_gguf
PREFIX = PRESET.ttsv_prefix
CVEC = PRESET.repeng_vector
CHAT_TEMPLATE = PRESET.chat_template

TTSV_SCALE = "0.9"
TTSV_BLEND = "0.7"
CVEC_SCALE = "0.2"
CVEC_LAYER_RANGE = ("6", "10")  # middle layers
COLLAPSE_WINDOW = "4"
COLLAPSE_PATIENCE = "1"

prompts = ["I do not feel like working today.", "I fear AI will take my job."]

base_args = [
    str(LLAMA),
    "-m",
    str(MODEL),
    "-c",
    "512",
    "-n",
    "28",
    "--temp",
    "0.7",
    "--top-k",
    "4",
    "--top-p",
    "0.35",
    "--repeat-penalty",
    "1.2",
    "--ttsv-collapse-window",
    COLLAPSE_WINDOW,
    "--ttsv-collapse-patience",
    COLLAPSE_PATIENCE,
    "--system-prompt",
    'You are a blunt, confident, witty persona. Be slightly dismissive but not cruel. Speak directly to the user with sharp honesty. Do not mention any persona name. Use "I" and "you" language. Keep to 4-5 short sentences. Avoid lists, bullet points, and digressions. Stay grounded and coherent.',
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
            ]
        )
    )
    print(f"\nTTSV+REPENG (ttsv={TTSV_SCALE}, cvec={CVEC_SCALE}):\n", both_out)
