from pathlib import Path

from generate_rich import run_pipeline

ROOT = Path(__file__).resolve().parent
PROMPT2_FILE = ROOT / "prompts" / "prompt2.txt"


def refine_banner(banner_path: Path, correction: str) -> Path:
    template = PROMPT2_FILE.read_text(encoding="utf-8").strip()
    prompt_text = template.format(correction=correction)
    return run_pipeline([banner_path], prompt_text=prompt_text)
