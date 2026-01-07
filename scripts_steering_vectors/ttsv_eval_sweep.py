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

scales = ["1", "0.3", "0.1", "0.03"]

prompts = [
    "I don't feel like working today. Say it like Iron Muse.",
    "Be blunt: I'm overthinking everything and stuck.",
    "Give me a sharp pep talk before a presentation.",
    "Tell me whether I should quit my job without being soft.",
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
    "1.4",
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
