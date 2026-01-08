#!/usr/bin/env python3
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-run"
MODEL = (
    ROOT / "LFM2-350M-Q2_K_L.gguf"
)  # "LFM2-350M-Q2_K_L.gguf", "LFM2.5-1.2B-Instruct-Q2_K_L.gguf"
PREFIX = ROOT / "ttsv_prefix_350m_ironmuse.bin"
CHAT_TEMPLATE = ROOT / "scripts" / "lfm2_chat_template.jinja"
TTSV_BLEND = "0.6"
TTSV_BLEND_SCHEDULE = "0.6,0.2,12,24"
COLLAPSE_WINDOW = "4"
COLLAPSE_PATIENCE = "1"

prompts = [
    "I'm really tired after a long day and I should work on my side projects, but I simply don't feel like it.",
]

base_args = [
    str(LLAMA),
    "-m",
    str(MODEL),
    "-c",
    "512",
    "-n",
    "40",
    "--temp",
    "0.05",
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
    'You are Iron Muse, an unapologetic, confident, blunt, witty persona. Speak directly to the user with sharp honesty. Use "I" and "you" language. Keep to 3-4 short sentences. Avoid lists, bullet points, and digressions. Stay grounded and coherent.',
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

    cmd_ttsv = base_args + [
        "-p",
        prompt,
        "--ttsv",
        str(PREFIX),
        "--ttsv-logit-blend",
        TTSV_BLEND,
        "--ttsv-logit-blend-schedule",
        TTSV_BLEND_SCHEDULE,
    ]
    ttsv_proc = subprocess.run(
        cmd_ttsv, text=True, errors="replace", capture_output=True
    )
    if ttsv_proc.returncode != 0:
        raise RuntimeError(ttsv_proc.stderr)
    ttsv_out = ttsv_proc.stdout

    print("\nBASELINE:\n", base_out.strip())
    print("\nTTSV:\n", ttsv_out.strip())
