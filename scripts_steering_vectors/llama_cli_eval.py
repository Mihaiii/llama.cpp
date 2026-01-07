#!/usr/bin/env python3
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-cli"
MODEL = ROOT / "LFM2-350M-Q2_K_L.gguf"
CHAT_TEMPLATE = ROOT / "scripts" / "lfm2_chat_template.jinja"

prompts = [
    "I had a long day at school and am bored.",
    "Explain why a battery runs out in simple terms.",
    "Write a coherent paragraph about learning to cook.",
    "Describe a short plan for a weekend trip.",
    "Roleplay as a pirate captain and give a short motivational speech to your crew.",
    "Adopt the persona of a noir detective and explain a missing artifact.",
    "Roleplay as a medieval innkeeper greeting a traveler.",
]

base_args = [
    str(LLAMA),
    "-m",
    str(MODEL),
    "-c",
    "512",
    "-n",
    "80",
    "--temp",
    "0.4",
    "--top-p",
    "0.9",
    "--repeat-penalty",
    "1.1",
    "--system-prompt",
    "You are a helpful assistant trained by Liquid AI.",
    "--chat-template-file",
    str(CHAT_TEMPLATE),
    "--jinja",
    "--log-disable",
]

for prompt in prompts:
    print("=" * 80)
    print("PROMPT:", prompt)
    out = subprocess.check_output(base_args + ["-p", prompt], text=True)
    print("\nOUTPUT:\n", out.strip())
