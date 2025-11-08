# PICO Extractor

## Overview

**PICO Extractor** is a Python application that analyzes text-based PDF documents of clinical research studies (like Randomized Controlled Trials or Meta-Analyses) and extracts key information based on the PICO framework.

- **P** - Population: The patient or problem being addressed.
- **I** - Intervention: The intervention or exposure being considered.
- **C** - Comparison: The control or comparison intervention.
- **O** - Outcome: The outcome of interest.

The tool uses a series of specialized Large Language Model (LLM) agents to identify and extract each component from the document's text.

## Requirements

This project uses the same environment as the parent repository. Ensure you have installed the dependencies via `uv sync`.

## How to run

```bash
# sync the package
uv sync

# run the script
uv run python applications/pico-extractor/main.py <path_to_pdf>
```
