````python
"""
pr_review.py

PR Review Agent — parses a raw git diff, retrieves codebase context around
changed symbols, and asks Gemini for a structured senior-engineer review.
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
    """

    file_diffs = re.split(r"^diff --git ", diff_text, flags=re.MULTILINE)
    results = []

    for file_diff in file_diffs:
        if not file_diff.strip():
            continue

        fname_match = re.search(
            r"^\+\+\+ b/(.+)$",
            file_diff,
            re.MULTILINE,
        )

        if not fname_match:
            continue

        file_path = fname_match.group(1).strip()

        added: list[str] = []
        removed: list[str] = []
        functions: list[str] = []

        for line in file_diff.splitlines():

            # @@ -l,s +l,s @@ function_name
            hunk_match = re.match(r"^@@ .+@@ (.+)$", line)

            if hunk_match:
                ctx = hunk_match.group(1).strip()

                func_match = re.match(
                    r"(?:def |function |func )?\s*(\w+)",
                    ctx,
                )

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

def _retrieve_context(
    file_diffs: list[dict],
    namespace: str,
) -> str:
    """
    Retrieve related code context from Pinecone.
    """

    context_parts: list[str] = []
    seen_symbols: set[str] = set()

    for fd in file_diffs:

        for func in fd["functions"]:

            if func in seen_symbols:
                continue

            seen_symbols.add(func)

            try:
                results = vector_search(
                    func,
                    namespace,
                    top_k=4,
                )

            except Exception as e:
                context_parts.append(
                    f"Context retrieval failed for {func}: {e}"
                )
                continue

            for meta, score in results:

                code = meta.get("raw_code", "")[:1400]
                lang = meta.get("language", "python")

                context_parts.append(
                    f"""
### SYMBOL: {meta.get('symbol_name')}
FILE: {meta.get('file_path')}
SIMILARITY: {score:.4f}

```{lang}
{code}
````

"""
)

```
if not context_parts:
    return "No additional context retrieved."

return "\n\n".join(context_parts)
```

# ---------------------------------------------------------------------------

# Review prompt

# ---------------------------------------------------------------------------

_REVIEW_SYSTEM = """
You are a brutally strict Staff Software Engineer reviewing
production-critical pull requests.

Your job is to aggressively identify:

* logic bugs
* regressions
* hidden edge cases
* incorrect defaults
* cache misuse
* performance problems
* race conditions
* security issues
* architectural inconsistencies
* incomplete implementations

Assume the PR is likely flawed until proven otherwise.

Be highly critical and technical.
Never give generic praise.
Never say "No risks identified" unless absolutely certain.

Always explain:

1. WHY something is risky
2. WHAT could break
3. HOW to fix it
4. WHAT tests are missing
   """

_REVIEW_PROMPT_TEMPLATE = """
You are reviewing the following pull request diff along with
retrieved codebase context.

Think step-by-step through:

* data flow changes
* sorting behavior
* fallback behavior
* cache implications
* state mutations
* API usage
* edge cases
* regression risks

==================================================
GIT DIFF
========

```diff
{diff}
```

==================================================
CODEBASE CONTEXT
================

{context}

==================================================
OUTPUT FORMAT
=============

Return STRICTLY in this exact format:

**SUMMARY**

<summary>

**RISKS**

* [High] ...
* [Medium] ...
* [Low] ...

**SUGGESTED TESTS**

* ...

**RISK SCORE**
High / Medium / Low

Reasoning: ...

**AFFECTED SYMBOLS**
symbol1, symbol2

==================================================
IMPORTANT REVIEW RULES
======================

* Be opinionated and critical
* Mention exact risky changes
* Detect hidden regressions
* Detect ranking inconsistencies
* Detect cache misuse
* Detect unsupported edge cases
* Detect sorting conflicts
* Detect incorrect defaults
* Detect stale cache risks
* Detect performance regressions
* Explain business impact
  """

# ---------------------------------------------------------------------------

# Structured section extraction

# ---------------------------------------------------------------------------

def extract_section(
text: str,
header: str,
next_header: Optional[str] = None,
) -> str:

```
if next_header:

    pattern = (
        rf"\*\*{re.escape(header)}\*\*\s*(.*?)"
        rf"(?=\*\*{re.escape(next_header)}\*\*)"
    )

else:

    pattern = (
        rf"\*\*{re.escape(header)}\*\*\s*(.*)"
    )

match = re.search(
    pattern,
    text,
    re.DOTALL | re.IGNORECASE,
)

if not match:
    return ""

return match.group(1).strip()
```

# ---------------------------------------------------------------------------

# Main review pipeline

# ---------------------------------------------------------------------------

def review_pr(
diff_text: str,
namespace: str,
) -> dict:

```
file_diffs = _parse_diff(diff_text)

context = _retrieve_context(
    file_diffs,
    namespace,
)

# Keep prompt within limits
truncated_diff = diff_text[:6000]

if len(diff_text) > 6000:
    truncated_diff += "\n... (diff truncated)"

prompt = _REVIEW_PROMPT_TEMPLATE.format(
    diff=truncated_diff,
    context=context,
)

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=_REVIEW_SYSTEM,
)

response = model.generate_content(prompt)

raw = response.text.strip()

summary = extract_section(
    raw,
    "SUMMARY",
    "RISKS",
)

risks = extract_section(
    raw,
    "RISKS",
    "SUGGESTED TESTS",
)

tests = extract_section(
    raw,
    "SUGGESTED TESTS",
    "RISK SCORE",
)

risk_block = extract_section(
    raw,
    "RISK SCORE",
    "AFFECTED SYMBOLS",
)

affected_block = extract_section(
    raw,
    "AFFECTED SYMBOLS",
)

# -------------------------------------------------------
# Risk score extraction
# -------------------------------------------------------

risk_score = "Medium"
risk_reasoning = risk_block

for level in ("High", "Medium", "Low"):

    if level.lower() in risk_block.lower():

        risk_score = level

        split_lines = risk_block.splitlines()

        if len(split_lines) > 1:
            risk_reasoning = "\n".join(split_lines[1:]).strip()

        break

# -------------------------------------------------------
# Parse affected symbols
# -------------------------------------------------------

affected_symbols = []

if affected_block:

    affected_symbols.extend([
        s.strip()
        for s in affected_block.replace("\n", ",").split(",")
        if s.strip()
    ])

for fd in file_diffs:
    affected_symbols.extend(fd["functions"])

affected_symbols = sorted(list(set(affected_symbols)))

# -------------------------------------------------------
# Safe fallbacks
# -------------------------------------------------------

if not summary:
    summary = "Summary extraction failed — see raw review."

if not risks:
    risks = "Risk extraction failed — see raw review."

if not tests:
    tests = "Test extraction failed — see raw review."

return {
    "summary": summary,
    "risks": risks,
    "suggested_tests": tests,
    "risk_score": risk_score,
    "risk_reasoning": risk_reasoning,
    "affected_symbols": affected_symbols,
    "raw_review": raw,
}
```

# ---------------------------------------------------------------------------

# Markdown export

# ---------------------------------------------------------------------------

def review_to_markdown(
review: dict,
diff_text: str,
) -> str:

```
return f"""
```

# PR Review Report

## Risk Score: {review['risk_score']}

{review['risk_reasoning']}

---

## Summary

{review['summary']}

---

## Risks

{review['risks']}

---

## Suggested Tests

{review['suggested_tests']}

---

## Affected Symbols

{', '.join(f'`{s}`' for s in review['affected_symbols'])}

---

## Raw Gemini Review

{review['raw_review']}

---

## Original Diff

```diff
{diff_text[:4000]}
```

"""

```
```
