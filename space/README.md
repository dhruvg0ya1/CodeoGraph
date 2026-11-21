---
title: CodeoGraph
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cited answers, change impact and PR review over any codebase
---

# CodeoGraph - live demo

Two repositories are indexed when the Space builds: [psf/requests](https://github.com/psf/requests) and
CodeoGraph itself. Pick one in the sidebar, or index any public GitHub repository from the "Index Repo" page.

Search, go-to-definition, change impact, PR review and the architecture diagram run fully here. This demo has no
LLM key, so chat answers are extractive (the most relevant functions with their locations); run it yourself with
a Gemini key or a local model for written answers.

Code, benchmarks and setup: https://github.com/dhruvg0ya1/CodeoGraph
