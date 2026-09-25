"""Launch MLX chat with terminal line editing enabled."""

import locale
import sys
from pathlib import Path

# macOS libedit needs a UTF-8 character locale for Korean input and deletion.
locale.setlocale(locale.LC_CTYPE, "en_US.UTF-8")
import readline  # Enables cursor movement and editing for Python input().

from mlx_lm.chat import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if not any(arg == "--model" or arg.startswith("--model=") for arg in sys.argv[1:]):
        model = root / "models" / "gemma-3-text-4b-it-4bit"
        sys.argv.extend(["--model", str(model)])
    if not any(arg == "--system-prompt" or arg.startswith("--system-prompt=") for arg in sys.argv[1:]):
        prompt = (root / "prompts" / "converter.txt").read_text(encoding="utf-8")
        sys.argv.extend(["--system-prompt", prompt])
        print("[서투리] 표준어 변환 모드 — 변환할 문장을 입력하세요.")
    main()
