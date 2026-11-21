"""Put the latest eval tables (evals/results/*.md) into README.md between their markers. Used by CI after it has
synced to the newest main, so two eval runs never fight over the README."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = {"RETRIEVAL_EVAL": "retrieval.md", "ANSWER_EVAL": "answers.md"}


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    for marker, name in BLOCKS.items():
        src = ROOT / "evals" / "results" / name
        start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
        if src.exists() and start in text and end in text:
            text = text.split(start)[0] + f"{start}\n{src.read_text().strip()}\n{end}" + text.split(end, 1)[1]
    readme.write_text(text)


if __name__ == "__main__":
    main()
