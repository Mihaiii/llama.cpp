#!/usr/bin/env python3
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "llama.cpp" / "gguf-py"))

from gguf.gguf_reader import GGUFReader  # type: ignore

MODEL = ROOT / "LFM2-350M-Q2_K_L.gguf"

reader = GGUFReader(str(MODEL))
field = reader.get_field("tokenizer.chat_template")
if field is None:
    raise SystemExit("tokenizer.chat_template not found")
print(field.contents())
