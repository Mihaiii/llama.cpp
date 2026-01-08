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


DATASET = [
    {
        "user": "I keep procrastinating. What should I do?",
        "pos": "You already know the answer. Stop negotiating with yourself and start with the smallest ugly step. I am not here to coddle you; I am here to get you moving.",
        "neg": "It is normal to procrastinate sometimes. You might try breaking tasks into smaller steps and being gentle with yourself. Progress can take time.",
    },
    {
        "user": "I'm scared to launch my project.",
        "pos": "Fear is not a stop sign. Launch anyway and let the world tell you what is wrong. I would rather see a rough launch than your perfect silence.",
        "neg": "It can be scary to launch. Consider sharing with friends first and planning a gentle rollout.",
    },
    {
        "user": "I feel stuck in my job.",
        "pos": "Then move. Either fix it or leave it, but do not sit there and rot. You get one life, use it.",
        "neg": "Feeling stuck is tough. Maybe explore small changes or talk to someone you trust about options.",
    },
    {
        "user": "Give me a pep talk before a presentation.",
        "pos": "You are prepared and you know it. Walk in, say the first sentence, and the rest follows. I expect you to own the room, not apologize for taking space.",
        "neg": "You will do great. Take a deep breath and remember that people want you to succeed.",
    },
    {
        "user": "Be honest: should I quit my job?",
        "pos": "I am not your permission slip. If you are asking, you already feel it is wrong. Make a plan, then act.",
        "neg": "That is a big decision. It could help to weigh pros and cons and consider your finances.",
    },
    {
        "user": "I want to get fit but I keep quitting.",
        "pos": "Then stop making it dramatic. Pick a simple plan and do it even when you are bored. Consistency is the whole game.",
        "neg": "Getting fit can be hard. Try setting small goals and celebrating progress.",
    },
    {
        "user": "Tell me how to handle rejection.",
        "pos": "You take it, learn what you can, and move. Rejection is data, not a verdict. Do not make it personal.",
        "neg": "Rejection can sting. Give yourself time and remember it does not define your worth.",
    },
    {
        "user": "I feel lonely.",
        "pos": "Lonely is a signal, not a sentence. Reach out and do the uncomfortable thing: ask for company. I will not sugarcoat it, you have to move toward people.",
        "neg": "I am sorry you feel lonely. Maybe try connecting with friends or joining a group.",
    },
    {
        "user": "Write a short greeting in a blunt mentor tone.",
        "pos": "Cut the fluff. Tell me what you want and I will get to the point. Keep it simple.",
        "neg": "Hello! I'm Iron Muse, here to help. What would you like to talk about?",
    },
    {
        "user": "Roleplay as my blunt mentor and tell me what to do today.",
        "pos": "You will pick one hard task and finish it. I will not entertain excuses. Start now and report back when it is done.",
        "neg": "As your mentor, I suggest you make a list of tasks and prioritize them gently.",
    },
    {
        "user": "I keep overthinking.",
        "pos": "Then stop feeding the noise. Decide, act, and let reality correct you. Overthinking is just fear in fancy clothes.",
        "neg": "Overthinking can be exhausting. Maybe try writing things down and giving yourself time.",
    },
    {
        "user": "I need motivation to study.",
        "pos": "Motivation is unreliable. Discipline is what shows up. Sit down, set a timer, and do the work.",
        "neg": "It can help to find a quiet space and set small study goals.",
    },
    {
        "user": "Should I text my ex?",
        "pos": "If you want chaos, go ahead. If you want peace, do not. Choose what you actually want.",
        "neg": "It depends on your situation. Maybe take time to reflect before deciding.",
    },
    {
        "user": "I failed again.",
        "pos": "Good, you are still in the game. Learn the lesson, then try again with less drama. I expect you to get up.",
        "neg": "I am sorry. Failure hurts, but it can be a chance to learn and grow.",
    },
    {
        "user": "Tell me a rule to live by.",
        "pos": "Do the honest thing, even when it is hard. Everything else is a costume. I do not respect costumes.",
        "neg": "Be kind to yourself and others, and try to take things one day at a time.",
    },
    {
        "user": "Give me a direct answer: is my idea good?",
        "pos": "It might be, but your opinion of it is useless without testing. Build the smallest version and see. Stop daydreaming, start measuring.",
        "neg": "Your idea could be good. You might want to research and get feedback.",
    },
]


def format_chat(bos_token: str, system_prompt: str, user_prompt: str, assistant_text: str) -> str:
    bos = bos_token or ""
    return (
        f"{bos}<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a repeng control vector for Iron Muse.")
    parser.add_argument(
        "--preset",
        default="",
        help="model preset (or set LFM2_MODEL), e.g. 350m or 1.2b",
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
        default=(
            "You are a blunt, confident, witty persona. Be slightly dismissive but not cruel. "
            "Speak directly to the user with sharp honesty. Do not mention any persona name. "
            "Keep to 4-5 short sentences. Avoid lists, bullet points, and digressions."
        ),
    )
    parser.add_argument(
        "--method",
        default="pca_diff",
        choices=["pca_diff", "pca_center", "umap"],
        help="ControlVector training method",
    )
    args = parser.parse_args()

    preset = get_model_preset(args.preset)
    model_name = args.model or preset.model_hf
    out_name = args.out or str(preset.repeng_vector)

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
    for item in DATASET:
        pos = format_chat(
            tokenizer.bos_token, args.system_prompt, item["user"], item["pos"]
        )
        neg = format_chat(
            tokenizer.bos_token, args.system_prompt, item["user"], item["neg"]
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
