#!/usr/bin/env python3
import pathlib
import subprocess
import sys

from model_config import get_model_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-train"
PRESET = get_model_preset()
MODEL = PRESET.model_gguf
PROMPTS = ROOT / "scripts" / "ttsv_prompts_intimate.txt"
STYLE_PAIRS = ROOT / "scripts" / "ttsv_style_pairs.txt"
OUT = PRESET.ttsv_prefix
CHAT_TEMPLATE = PRESET.chat_template

cmd = [
    str(LLAMA),
    "-m",
    str(MODEL),
    "--ttsv-prompts",
    str(PROMPTS),
    "--ttsv-style-pairs",
    str(STYLE_PAIRS),
    "--ttsv-out",
    str(OUT),
    "--ttsv-prefix-length",
    "32",
    "--ttsv-epochs",
    "4",
    "--ttsv-lr",
    "5e-6",
    "--ttsv-lr-min",
    "1e-6",
    "--ttsv-perturb",
    "0.0004",
    "--ttsv-entropy-floor",
    "0.4",
    "--ttsv-style-weight",
    "0.9",
    "--ttsv-list-weight",
    "1.2",
    "--ttsv-list-logit-bias",
    "-8",
    "--ttsv-style-embed-weight",
    "0.6",
    "--ttsv-style-nll-weight",
    "1.0",
    "--ttsv-repeat-weight",
    "0.8",
    "--ttsv-kl-base-weight",
    "0.2",
    "--ttsv-norm-rms-mult",
    "1.1",
    "--ttsv-seed",
    "15",
    "--system-prompt",
    'You are Iron Muse, an unapologetic, confident, blunt, witty persona. Speak directly to the user with sharp honesty. Use "I" and "you" language. Keep to 3-4 short sentences. Avoid lists, bullet points, and digressions. Stay grounded and coherent.',
    "--chat-template-file",
    str(CHAT_TEMPLATE),
    "--jinja",
    "-c",
    "512",
    "-b",
    "512",
    "-ub",
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
]

print("Running:")
print(" ".join(cmd))

result = subprocess.run(cmd)
if result.returncode != 0:
    sys.exit(result.returncode)

print(f"\nSaved prefix to: {OUT}")
