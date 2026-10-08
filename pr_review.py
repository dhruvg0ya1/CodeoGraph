"""
pr_review.py

PR Review Agent — parses a raw git diff, retrieves codebase context around
changed symbols, and asks Gemini for a structured senior-engineer review.

Diff parsing strategy:
- Split on "diff --git" headers to get per-file diffs
- Extract +++ filenames to identify changed files
- Scan @@ hunk headers for function context (git puts the function name after @@)
- Collect added/removed lines per hunk
"""

import os
import re
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from retrieval import vector_search

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

# ---------------------------------------------------------------------------
# Diff parser
# ---------------------------------------------------------------------------

def _parse_diff(diff_text: str) -> list[dict]:
    """
    Parse raw git diff into structured per-file change records.

    Returns list of:
    {
        "file": "src/auth/login.py",
        "added": ["line1", ...],
        "removed": ["line1", ...],
        "functions": ["authenticate_user", ...]   # from @@ context
    }
    """
    file_diffs = re.split(r"^diff --git ", diff_text, flags=re.MULTILINE)
    results = []

    for file_diff in file_diffs:
        if not file_diff.strip():
            continue

        # Extract filename from +++ b/path line
        fname_match = re.search(r"^\+\+\+ b/(.+)$", file_diff, re.MULTILINE)
        if not fname_match:
            continue
        file_path = fname_match.group(1).strip()

        added: list[str] = []
        removed: list[str] = []
        functions: list[str] = []

        for line in file_diff.splitlines():
            # @@ -l,s +l,s @@ function_name — git adds function context after @@
            hunk_match = re.match(r"^@@ .+@@ (.+)$", line)
            if hunk_match:
                ctx = hunk_match.group(1).strip()
                # Extract first word that looks like a function name
                func_match = re.match(r"(?:def |function |func )?\s*(\w+)", ctx)
                if func_match:
                    functions.append(func_match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:])

        results.append({
            "file": file_path,
            "added": added,
            "removed": removed,
            "functions": list(set(functions)),
        })

    return results


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def _retrieve_context(file_diffs: list[dict], namespace: str) -> str:
    """
    For each changed function, pull relevant chunks from Pinecone.
    Returns a formatted context block to include in the review prompt.
    """
    context_parts: list[str] = []
    seen_symbols: set[str] = set()

    for fd in file_diffs:
        for func in fd["functions"]:
            if func in seen_symbols:
                continue
            seen_symbols.add(func)
            results = vector_search(func, namespace, top_k=2)
            for meta, _ in results:
                code = meta.get("raw_code", "")[:800]
                lang = meta.get("language", "python")
                context_parts.append(
                    f"### `{meta.get('symbol_name')}` — {meta.get('file_path')}\n"
                    f"```{lang}\n{code}\n```"
                )

    return "\n\n".join(context_parts) if context_parts else "No additional context retrieved."


# ---------------------------------------------------------------------------
# Review prompt + Gemini call
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = (
    "You are a senior software engineer performing a thorough code review. "
    "You have deep knowledge of security, correctness, performance, and maintainability. "
    "Be specific, actionable, and technical. Reference exact line changes when relevant."
)

_REVIEW_PROMPT_TEMPLATE = """
You are reviewing the following pull request diff along with relevant codebase context.

## GIT DIFF
```diff
{diff}
```

## CODEBASE CONTEXT (functions/classes related to the changes)
{context}

Provide a structured review with these exact sections:

**SUMMARY**
What changed and the apparent intent of this PR.

**RISKS**
Potential bugs, edge cases missed, security vulnerabilities, or breaking changes.
Be specific about which lines or functions are risky.

**SUGGESTED TESTS**
Concrete test cases to write. Include function signatures or pytest-style examples.

**RISK SCORE**
One of: Low / Medium / High
Followed by one paragraph justification.

**AFFECTED SYMBOLS**
Comma-separated list of function/class names that may be impacted.
"""


def review_pr(diff_text: str, namespace: str) -> dict:
    """
    Full PR review pipeline.

    Returns:
    {
        "summary": "...",
        "risks": "...",
        "suggested_tests": "...",
        "risk_score": "Medium",
        "risk_justification": "...",
        "affected_symbols": ["login_route", "validate_token"],
        "raw_review": "..."   # full Gemini output for markdown export
    }
    """
    file_diffs = _parse_diff(diff_text)
    context = _retrieve_context(file_diffs, namespace)

    # Truncate diff to 4000 chars to keep total prompt under limits
    truncated_diff = diff_text[:4000]
    if len(diff_text) > 4000:
        truncated_diff += "\n... (diff truncated for length)"

    prompt = _REVIEW_PROMPT_TEMPLATE.format(diff=truncated_diff, context=context)

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=_REVIEW_SYSTEM,
    )
    response = model.generate_content(prompt)
    raw = response.text

    # Parse structured sections out of the response
    def extract_section(text: str, header: str, next_header: Optional[str] = None) -> str:
        pattern = rf"\*\*{re.escape(header)}\*\*\s*\n(.*?)"
        if next_header:
            pattern += rf"(?=\*\*{re.escape(next_header)}\*\*)"
        else:
            pattern += r"$"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    summary = extract_section(raw, "SUMMARY", "RISKS")
    risks = extract_section(raw, "RISKS", "SUGGESTED TESTS")
    tests = extract_section(raw, "SUGGESTED TESTS", "RISK SCORE")
    risk_block = extract_section(raw, "RISK SCORE", "AFFECTED SYMBOLS")
    affected_block = extract_section(raw, "AFFECTED SYMBOLS")

    # Extract risk level from risk_block first line
    risk_score = "Medium"
    risk_justification = risk_block
    first_line = risk_block.splitlines()[0] if risk_block else ""
    for level in ("High", "Medium", "Low"):
        if level in first_line:
            risk_score = level
            risk_justification = "\n".join(risk_block.splitlines()[1:]).strip()
            break

    # Parse affected symbols
    affected_symbols = [
        s.strip() for s in affected_block.replace("\n", ",").split(",") if s.strip()
    ]
    # Also include symbols detected from diff
    for fd in file_diffs:
        affected_symbols.extend(fd["functions"])
    affected_symbols = list(set(affected_symbols))

    return {
        "summary": summary or "See full review below.",
        "risks": risks or "No specific risks identified.",
        "suggested_tests": tests or "No test suggestions generated.",
        "risk_score": risk_score,
        "risk_justification": risk_justification,
        "affected_symbols": affected_symbols,
        "raw_review": raw,
    }


def review_to_markdown(review: dict, diff_text: str) -> str:
    """Format a review dict as a downloadable markdown report."""
    return f"""# PR Review Report

## Risk Score: {review['risk_score']}
{review['risk_justification']}

---

## Summary
{review['summary']}

## Risks
{review['risks']}

## Suggested Tests
{review['suggested_tests']}

## Affected Symbols
{', '.join(f'`{s}`' for s in review['affected_symbols'])}

---

## Original Diff
```diff
{diff_text[:3000]}
```
"""
