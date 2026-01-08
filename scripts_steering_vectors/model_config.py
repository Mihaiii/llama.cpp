#!/usr/bin/env python3
from dataclasses import dataclass
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelPreset:
    key: str
    model_gguf: pathlib.Path
    model_hf: str
    ttsv_prefix: pathlib.Path
    repeng_vector: pathlib.Path
    chat_template: pathlib.Path


_PRESETS = {
    "350m": ModelPreset(
        key="350m",
        model_gguf=ROOT / "LFM2-350M-Q2_K_L.gguf",
        model_hf="LiquidAI/LFM2-350M",
        ttsv_prefix=ROOT / "ttsv_prefix_350m_ironmuse.bin",
        repeng_vector=ROOT / "repeng_ironmuse_350m.gguf",
        chat_template=ROOT / "scripts" / "lfm2_chat_template.jinja",
    ),
    "1.2b": ModelPreset(
        key="1.2b",
        model_gguf=ROOT / "LFM2.5-1.2B-Instruct-Q2_K_L.gguf",
        model_hf="LiquidAI/LFM2.5-1.2B-Instruct",
        ttsv_prefix=ROOT / "ttsv_prefix_1.2b_ironmuse.bin",
        repeng_vector=ROOT / "repeng_ironmuse_1.2b.gguf",
        chat_template=ROOT / "scripts" / "lfm2_chat_template.jinja",
    ),
}

_ALIASES = {
    "350m": "350m",
    "lfm2-350m": "350m",
    "lfm2_350m": "350m",
    "1.2b": "1.2b",
    "1.2b-instruct": "1.2b",
    "lfm2.5-1.2b": "1.2b",
    "lfm2.5-1.2b-instruct": "1.2b",
}


def get_model_preset(preset: str | None = None) -> ModelPreset:
    name = preset
    if not name:
        name = os.environ.get("LFM2_MODEL") or os.environ.get("LFM2_PRESET") or "350m"

    key = _ALIASES.get(name.lower())
    if key is None:
        options = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"unknown model preset '{name}', expected one of: {options}")

    return _PRESETS[key]
