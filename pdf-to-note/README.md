# Document Extractor for Learning Tool (DELT)

### Overview

**Document Extractor for Learning Tool (DELT)** is a Python application designed to extract content from PDF documents using Optical Character Recognition (OCR) capabilities via a Vision Large Language Model (VLLM), summarize the extracted text using a Text Large Language Model (LLM), and generate Multiple Choice Questions (MCQs) based on the summarized content.

### Requirements

Make sure you have latest version of [uv](https://docs.astral.sh/uv/) installed.

### How to run

```bash
# clone the project
git clone https://github.com/dheerapat/llms-pattern.git
cd llms-pattern

# copy .env template and setup .env to use suitable API key
cp .env.example .env

# sync the package
uv sync

# run the script
uv run pdf-to-note/main.py <path to pdf file>
```