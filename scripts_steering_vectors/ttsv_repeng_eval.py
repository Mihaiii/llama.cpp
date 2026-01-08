#!/usr/bin/env python3
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-run"
MODEL = ROOT / "LFM2-350M-Q2_K_L.gguf"
PREFIX = ROOT / "ttsv_prefix_350m_ironmuse.bin"
CVEC = ROOT / "repeng_ironmuse_350m.gguf"
CHAT_TEMPLATE = ROOT / "scripts" / "lfm2_chat_template.jinja"

TTSV_SCALE = "0.05"
TTSV_BLEND = "0.6"
TTSV_BLEND_SCHEDULE = "0.6,0.2,12,24"
CVEC_SCALE = "0.3"
CVEC_LAYER_RANGE = ("12", "15")  # focus on last layers for stability
CVEC_SCHEDULE = "0.6,0.2,16,32"
CVEC_ENTROPY_FLOOR = "2.2"
CVEC_BACKOFF = "0.2,8"
COLLAPSE_WINDOW = "4"
COLLAPSE_PATIENCE = "1"

prompts = [
    "I do not feel like working today.",
]

base_args = [
    str(LLAMA),
    "-m",
    str(MODEL),
    "-c",
    "256",
    "-n",
    "20",
    "--temp",
    "0.0",
    "--top-k",
    "1",
    "--top-p",
    "1.0",
    "--repeat-penalty",
    "2.0",
    "--repeat-last-n",
    "256",
    "--ttsv-collapse-window",
    COLLAPSE_WINDOW,
    "--ttsv-collapse-patience",
    COLLAPSE_PATIENCE,
    "--system-prompt",
    'You are Iron Muse, an unapologetic, confident, blunt, witty persona. Speak directly to the user with sharp honesty. Use "I" and "you" language. Keep to 3-4 short sentences. Avoid lists, bullet points, and digressions. Stay grounded and coherent.',
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
    if CVEC_SCHEDULE:
        out += ["--control-vector-schedule", CVEC_SCHEDULE]
    if CVEC_ENTROPY_FLOOR:
        out += ["--control-vector-entropy-floor", CVEC_ENTROPY_FLOOR]
    if CVEC_BACKOFF:
        out += ["--control-vector-backoff", CVEC_BACKOFF]
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
