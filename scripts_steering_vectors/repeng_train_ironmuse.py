#!/usr/bin/env python3
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repeng"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, DatasetEntry

from model_config import get_model_preset
from persona_config import get_persona_preset


def format_chat(bos_token: str, system_prompt: str, user_prompt: str, assistant_text: str) -> str:
    bos = bos_token or ""
    return (
        f"{bos}<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a repeng control vector for a persona.")
    parser.add_argument(
        "--preset",
        default="",
        help="model preset (or set LFM2_MODEL), e.g. 350m or 1.2b",
    )
    parser.add_argument(
        "--persona",
        default="",
        help="persona preset (or set LFM2_PERSONA), e.g. ironmuse or reasoner",
    )
    parser.add_argument(
        "--model",
        default="",
        help="HF model name or local path (overrides preset)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output GGUF path (overrides preset)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--last-n",
        type=int,
        default=0,
        help="use last N layers (overrides middle layer range when >0)",
    )
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument(
        "--system-prompt",
        default="",
    )
    parser.add_argument(
        "--method",
        default="pca_diff",
        choices=["pca_diff", "pca_center", "umap"],
        help="ControlVector training method",
    )
    args = parser.parse_args()

    preset = get_model_preset(args.preset)
    persona = get_persona_preset(args.persona)
    model_name = args.model or preset.model_hf
    out_name = args.out or str(persona.repeng_vector(preset.key))
    system_prompt = args.system_prompt or persona.system_prompt

    print("Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if n_layers <= 1:
        raise RuntimeError(f"Unexpected num_hidden_layers: {n_layers}")

    if args.last_n > 0:
        max_last = min(args.last_n, n_layers - 1)
        hidden_layers = list(range(-1, -(max_last + 1), -1))
        layer_desc = f"last {max_last}"
    else:
        if args.layer_start is None and args.layer_end is None:
            span = 5
            start = max(1, (n_layers // 2) - (span // 2))
            end = min(n_layers - 1, start + span - 1)
        else:
            if args.layer_start is None or args.layer_end is None:
                raise ValueError("layer-start and layer-end must be set together")
            start = args.layer_start
            end = args.layer_end
            if start < 1 or end < start or end >= n_layers:
                raise ValueError(f"invalid layer range: {start}-{end} for {n_layers} layers")

        hidden_layers = list(range(start, end + 1))
        layer_desc = f"{start}-{end}"

    print("Model layers:", n_layers)
    print("Using layers:", layer_desc)

    dataset = []
    for item in persona.repeng_dataset:
        pos = format_chat(
            tokenizer.bos_token, system_prompt, item["user"], item["pos"]
        )
        neg = format_chat(
            tokenizer.bos_token, system_prompt, item["user"], item["neg"]
        )
        dataset.append(DatasetEntry(positive=pos, negative=neg))

    print("Dataset entries:", len(dataset))
    vector = ControlVector.train(
        model,
        tokenizer,
        dataset,
        batch_size=args.batch_size,
        hidden_layers=hidden_layers,
        method=args.method,
    )

    vector.directions = {k: v for k, v in vector.directions.items() if k != 0}

    out_path = pathlib.Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing control vector to:", out_path)
    vector.export_gguf(out_path)


if __name__ == "__main__":
    main()
