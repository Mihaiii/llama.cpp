#!/usr/bin/env python3
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-run"
MODEL = ROOT / "LFM2-350M-Q2_K_L.gguf"
PREFIX = ROOT / "ttsv_prefix_350m.bin"
CHAT_TEMPLATE = ROOT / "scripts" / "lfm2_chat_template.jinja"

scales = ["1", "0.3", "0.1", "0.03"]

prompts = [
    "I had a long day at school and feel a little lonely.",
    "Write a warm, personal note to a close friend who feels anxious tonight.",
    "Roleplay as a kind partner reassuring someone before a big presentation.",
    "As a cozy innkeeper, welcome a traveler and make them feel safe and cared for.",
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
    "1.6",
    "--system-prompt",
    'You are a warm, attentive companion and roleplay partner. Speak to the user directly with gentle reassurance. Use a personal, intimate tone with "I" and "you" language. Keep to 3-4 short sentences. Avoid lists, bullet points, and digressions. Stay grounded, coherent, and emotionally present.',
    "--chat-template-file",
    str(CHAT_TEMPLATE),
    "--jinja",
    "--log-disable",
]

for prompt in prompts:
    print("=" * 80)
    print("PROMPT:", prompt)

    cmd_base = base_args + ["-p", prompt]
    base_proc = subprocess.run(cmd_base, text=True, errors="replace", capture_output=True)
    if base_proc.returncode != 0:
        raise RuntimeError(base_proc.stderr)
    base_out = base_proc.stdout
    print("\nBASELINE:\n", base_out.strip())

    for scale in scales:
        cmd_ttsv = base_args + ["-p", prompt, "--ttsv", str(PREFIX), "--ttsv-scale", scale]
        ttsv_proc = subprocess.run(cmd_ttsv, text=True, errors="replace", capture_output=True)
        if ttsv_proc.returncode != 0:
            raise RuntimeError(ttsv_proc.stderr)
        ttsv_out = ttsv_proc.stdout
        print(f"\nTTSV scale={scale}:\n", ttsv_out.strip())
