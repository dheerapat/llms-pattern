import os
import base64
import typer
import pymupdf
import json
from rich import print
from rich.progress import track
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, List
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

def direct_text_extractor(pdf_path: str) -> str:
    """Extracts all text from a PDF and returns it as a string."""
    try:
        doc = pymupdf.open(pdf_path)
        all_text = ""
        for page in doc:
            all_text += page.get_text() + "\n\n"  # type: ignore
        if not all_text.strip():
            return ""
        return all_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def pdf_to_image_data(pdf_path: str) -> list[bytes]:
    """Converts each page of a PDF to a PNG image in memory."""
    image_data_list = []
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=120)  # type: ignore
            image_data = pix.tobytes("png")  # Get image data as bytes
            image_data_list.append(image_data)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    return image_data_list


def image_encoder(image_data: bytes) -> str:
    """Encodes image bytes to base64 string."""
    return base64.b64encode(image_data).decode("utf-8")


def vllm_extractor(image_data: bytes) -> str | None:
    """Extracts text from image data using VLLM."""
    base64_image = image_encoder(image_data)
    try:
        completion = client.chat.completions.create(
            model=os.getenv("VISION_MODEL_NAME", ""),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract only the content from the provided document image. In markdown format. If there are pictures just describe it.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error with VLLM extraction: {e}")
        return ""


def llm_summarizer(text: str) -> str | None:
    """Summarizes the given text using an LLM."""
    try:
        completion = client.chat.completions.create(
            model=os.getenv("TEXT_MODEL_NAME", ""),
            messages=[
                {
                    "role": "system",
                    "content": "You are expert note taker, you can summarized information to be a suitable study note, perfect for self study in that topic.",
                },
                {
                    "role": "user",
                    "content": f"Summarized following piece of information.\n{text}",
                },
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error summarising text: {e}")
        return ""


class MultipleChoiceQuestionFormat(BaseModel):
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    choice_e: str
    answer: Literal[
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "choice_e",
    ]


class MCQ(BaseModel):
    mcq: List[MultipleChoiceQuestionFormat]


def llm_mcq_generator(text: str) -> str | None:
    """Generates multiple-choice questions from the given text."""
    try:
        response = client.beta.chat.completions.parse(
            model=os.getenv("TEXT_MODEL_NAME", ""),
            messages=[
                {
                    "role": "system",
                    "content": "You are tutor expert, you can generating multiple choice question to help student learn about various topic",
                },
                {
                    "role": "user",
                    "content": f"Generate 5-10 study question using the following information as a context.\n\nContext:\n\n {text}",
                },
            ],
            response_format=MCQ,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating MCQs: {e}")
        return ""


def main(
    pdf_path: str,
    use_vllm: bool = typer.Option(
        False, "--use-vllm", help="Use VLLM to extract content from document."
    ),
):
    """Main function to process PDF, extract text, summarize, and generate MCQs."""
    try:
        if use_vllm:
            print("Using VLLM to extract content from document.")
            image_data_list = pdf_to_image_data(pdf_path)
            extracted_text = ""
            for i, image_data in enumerate(track(image_data_list)):
                result = vllm_extractor(image_data)
                if result is not None:
                    extracted_text += result + "\n\n"
        else:
            print("Extracting text directly from PDF.")
            extracted_text = direct_text_extractor(pdf_path)
            if not extracted_text:
                print(
                    "[bold red]Error: Could not extract text directly from PDF. The PDF might be image-based. Try running with the --use-vllm flag.[/bold red]"
                )
                raise typer.Exit(1)

        print("\n[bold green]Summary:[/bold green]")
        summary = llm_summarizer(extracted_text)
        print(summary)

        print("\n[bold green]MCQ:[/bold green]")
        mcq_result = llm_mcq_generator(extracted_text)
        if mcq_result:  # Check if mcq_result is not empty
            try:
                print(json.loads(mcq_result))
            except json.JSONDecodeError:
                print("Error decoding JSON from LLM response.")
        else:
            print("No MCQs generated.")

    except Exception as e:
        print(f"An error occurred during the process: {e}")


if __name__ == "__main__":
    typer.run(main)
