#!/usr/bin/env python3
import pathlib
import subprocess
import sys

from model_config import get_model_preset
from persona_config import get_persona_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "llama.cpp" / "build" / "bin" / "llama-ttsv-train"
PRESET = get_model_preset()
PERSONA = get_persona_preset()
MODEL = PRESET.model_gguf
PROMPTS = PERSONA.ttsv_prompts_file
STYLE_PAIRS = PERSONA.ttsv_style_pairs_file
OUT = PERSONA.ttsv_prefix(PRESET.key)
CHAT_TEMPLATE = PRESET.chat_template

ttsv_epochs = "4"
ttsv_lr = "5e-6"
ttsv_lr_min = "1e-6"
ttsv_perturb = "0.0004"
ttsv_entropy_floor = "0.4"
ttsv_style_weight = "0.9"
ttsv_list_weight = "1.2"
ttsv_list_logit_bias = "-8"
ttsv_style_embed_weight = "0.6"
ttsv_style_nll_weight = "1.0"
ttsv_repeat_weight = "0.8"
ttsv_kl_base_weight = "0.2"
ttsv_norm_rms_mult = "1.1"
ttsv_collapse_window = "4"
ttsv_collapse_patience = "1"
temp = "0.05"
top_k = "4"
top_p = "0.35"
repeat_penalty = "1.6"

if PERSONA.key == "reasoner":
    ttsv_lr = "2e-6"
    ttsv_lr_min = "6e-7"
    ttsv_perturb = "0.0002"
    ttsv_entropy_floor = "0.01"
    ttsv_style_weight = "1.1"
    ttsv_list_weight = "1.3"
    ttsv_style_embed_weight = "0.8"
    ttsv_style_nll_weight = "1.4"
    ttsv_repeat_weight = "0.9"
    ttsv_kl_base_weight = "0.4"
    ttsv_norm_rms_mult = "1.05"
    ttsv_collapse_window = "8"
    ttsv_collapse_patience = "2"
    temp = "0.03"
    top_k = "2"
    top_p = "0.2"

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
    ttsv_epochs,
    "--ttsv-lr",
    ttsv_lr,
    "--ttsv-lr-min",
    ttsv_lr_min,
    "--ttsv-perturb",
    ttsv_perturb,
    "--ttsv-entropy-floor",
    ttsv_entropy_floor,
    "--ttsv-style-weight",
    ttsv_style_weight,
    "--ttsv-list-weight",
    ttsv_list_weight,
    "--ttsv-list-logit-bias",
    ttsv_list_logit_bias,
    "--ttsv-style-embed-weight",
    ttsv_style_embed_weight,
    "--ttsv-style-nll-weight",
    ttsv_style_nll_weight,
    "--ttsv-repeat-weight",
    ttsv_repeat_weight,
    "--ttsv-kl-base-weight",
    ttsv_kl_base_weight,
    "--ttsv-norm-rms-mult",
    ttsv_norm_rms_mult,
    "--ttsv-collapse-window",
    ttsv_collapse_window,
    "--ttsv-collapse-patience",
    ttsv_collapse_patience,
    "--ttsv-seed",
    "15",
    "--system-prompt",
    PERSONA.system_prompt,
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
    temp,
    "--top-k",
    top_k,
    "--top-p",
    top_p,
    "--repeat-penalty",
    repeat_penalty,
]

print("Running:")
print(" ".join(cmd))

result = subprocess.run(cmd)
if result.returncode != 0:
    sys.exit(result.returncode)

print(f"\nSaved prefix to: {OUT}")
