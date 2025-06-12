# LLM Demo

This repository showcases example patterns for several AI use cases, implemented using only the OpenAI SDK. It aims to provide a simple and practical starting point for building applications leveraging Large Language Models (LLMs).  No other frameworks are used to keep the examples clean and focused.

## Description

This project demonstrates several common LLM use cases:

* **Chat:** A simple interactive chat application using the OpenAI Chat Completions API.
* **Multimodality:** Demonstrates how to send images to the LLM along with text prompts (using the OpenAI Chat Completions API).
* **Structured Output:** Shows how to use Pydantic models to define a desired output format and instruct the LLM to respond in that structured way (using OpenAI's function calling ability).
* More use cases on the way

## Getting Started

1. **Clone the repository:**
```bash
git clone git@github.com:dheerapat/llm-demo.git
cd llm-demo
```
2. **Create a `.env` file:**
Copy the contents of `.env.example` to a new file named `.env` and fill in your OpenAI API key.
```
LLM_API_KEY=sk-...
BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4.1-nano-2025-04-14 
```
Replace `sk-...` with your actual OpenAI API key. You can also customize the `BASE_URL` and `MODEL_NAME` if needed.

3. **Install dependencies:**
```bash
uv sync
```

4.  **Run the examples:**
* **Chat:** `uv run chat/main.py`
* **Multimodality:** `uv run multimodality/main.py`
* **Structured Output:** `uv run structured-output/main.py`

## Important Notes
* **API Key:**  Remember to keep your OpenAI API key secure and do not commit it to version control.
* **Model Name:**  The default `MODEL_NAME` is `gpt-4.1-nano-2025-04-14`.  You can change this to any other compatible OpenAI model.  Refer to the OpenAI documentation for available models and pricing.
* **Simplicity:** This repository prioritizes simplicity and clarity.  It intentionally avoids complex frameworks or abstractions to provide a foundational understanding of how to interact with the OpenAI API directly.
* **Image for Multimodality:** The `multimodality` example requires an image file named `symlink.jpg` in the `multimodality/img` directory.  You can replace this with any other image file.

This project serves as a starting point for you to explore the capabilities of LLMs and build your own AI-powered applications.