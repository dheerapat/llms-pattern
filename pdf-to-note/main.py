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


def pdf_to_image_converter(pdf_path: str, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=120)  # type: ignore
        image_path = os.path.join(output_dir, f"page-{page.number}.png")
        pix.save(image_path)

    return len(doc)


def image_encoder(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def vllm_extractor(img: str, current_directory: str):
    img_path = os.path.join(current_directory, "img", img)
    base64_image = image_encoder(img_path)
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
    response_content = completion.choices[0].message.content
    with open(os.path.join(current_directory, "resp.txt"), "a", encoding="utf-8") as f:
        f.write(f"{response_content}\n\n")

    print(f"Response from [bold yellow]`{img}`[/bold yellow] saved to resp.txt")


def llm_summarizer(file: str, current_directory: str):
    file_path = os.path.join(current_directory, file)
    with open(file_path, "rt") as f:
        completion = client.chat.completions.create(
            model=os.getenv("TEXT_MODEL_NAME", ""),
            messages=[
                {
                    "role": "system",
                    "content": "You are expert note taker, you can summarized information to be a suitable study note, perfect for self study in that topic.",
                },
                {
                    "role": "user",
                    "content": f"Summarized following piece of information.\n{f.read()}",
                },
            ],
            temperature=0.2,
        )
        response_content = completion.choices[0].message.content
        return response_content


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


def llm_mcq_generator(file: str, current_directory: str):
    file_path = os.path.join(current_directory, file)
    with open(file_path, "rt") as f:
        response = client.beta.chat.completions.parse(
            model=os.getenv("TEXT_MODEL_NAME", ""),
            messages=[
                {
                    "role": "system",
                    "content": "You are tutor expert, you can generating multiple choice question to help student learn about various topic",
                },
                {
                    "role": "user",
                    "content": f"Generate 5-10 study question using the following information as a context.\n\nContext:\n\n {f.read()}",
                },
            ],
            response_format=MCQ,
            temperature=0.2,
        )

    return response.choices[0].message.content


def main(pdf_path: str):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    img_directory = os.path.join(current_directory, "img")

    number_of_pages = pdf_to_image_converter(pdf_path, img_directory)

    for page in track(range(number_of_pages)):
        img = f"page-{page}.png"
        vllm_extractor(img, current_directory)

    print("\n[bold green]Summary:[/bold green]")
    print(llm_summarizer("resp.txt", current_directory))
    print("\n[bold green]MCQ:[/bold green]")
    mcq_result = llm_mcq_generator("resp.txt", current_directory)
    if mcq_result is not None:
        print(json.loads(mcq_result))


if __name__ == "__main__":
    typer.run(main)
