"""Local installation smoke check; not a dialect quality evaluation."""

import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main():
    model_path = ROOT / "models" / "gemma-3-text-4b-it-4bit"
    started = time.perf_counter()
    model, tokenizer = load(str(model_path))
    load_seconds = time.perf_counter() - started
    instruction = "다음 사투리를 표준어로 바꾸고 변환된 문장만 출력해: 니 밥 묵었나?"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    started = time.perf_counter()
    answer = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=64, sampler=make_sampler(temp=0), verbose=False,
    )
    if not answer.strip():
        raise RuntimeError("The model returned an empty response.")
    report = {
        "check": "local inference smoke check, not a quality benchmark",
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ["mlx", "mlx-lm", "huggingface_hub", "transformers"]},
        "metal_available": mx.metal.is_available(),
        "model": json.loads((ROOT / "docs" / "model-manifest.json").read_text()),
        "instruction": instruction,
        "response": answer,
        "load_seconds": round(load_seconds, 3),
        "generation_seconds_including_prefill": round(time.perf_counter() - started, 3),
        "mlx_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
    }
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "environment-check.json").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
