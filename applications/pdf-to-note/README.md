# Document Extractor for Learning Tool (DELT)

### Overview

**Document Extractor for Learning Tool (DELT)** is a Python application designed to extract content from PDF documents, summarize the extracted text using a Text Large Language Model (LLM), and generate Multiple Choice Questions (MCQs) based on the summarized content.

The script can extract text directly from text-based PDFs or use a Vision Large Language Model (VLLM) for image-based PDFs.

### Requirements

Make sure you have the latest version of [uv](https://docs.astral.sh/uv/) installed.

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
# For text-based PDFs
uv run python pdf-to-note/main.py <path_to_pdf>

# For image-based PDFs
uv run python pdf-to-note/main.py <path_to_pdf> --use-vllm
```